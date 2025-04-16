import torch
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_convert, box_iou
from DETR.lightning_module import DETRLightningModule
from typing import List, Dict
from tqdm import tqdm
import heapq


def run_detr_inference(model, image: torch.Tensor):
    """Handle DETR inference with proper input format and output parsing."""
    outputs = model([image])  # pass as list of tensors
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    probs = logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)  # skip "no object" class

    keep = scores > 0.3
    return {
        "boxes": boxes[keep].cpu(),
        "scores": scores[keep].cpu(),
        "labels": labels[keep].cpu(),
    }


@torch.inference_mode()
def evaluate_map_metrics(model: torch.nn.Module, dataloader) -> List[Dict]:
    model.eval()
    metric = MeanAveragePrecision()
    worst_heap = []
    is_detr = isinstance(model, DETRLightningModule)

    for idx, (images, targets) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating"
    ):
        try:
            images = [img.cuda() for img in images]
            outputs = model(images)

            for i in range(len(images)):
                if is_detr:
                    logits = outputs["pred_logits"][i]
                    boxes = outputs["pred_boxes"][i]

                    probs = logits.softmax(-1)
                    scores, labels = probs[..., :-1].max(-1)
                    keep = scores > 0.3

                    img_h, img_w = images[i].shape[-2:]
                    scale = torch.tensor(
                        [img_w, img_h, img_w, img_h], device=boxes.device
                    )
                    boxes_abs = (
                        box_convert(boxes[keep], in_fmt="cxcywh", out_fmt="xyxy")
                        * scale
                    )

                    pred_boxes_i = boxes_abs.cpu()
                    pred_scores_i = scores[keep].cpu()
                    pred_labels_i = labels[keep].cpu()
                else:
                    pred = outputs[i]
                    pred_boxes_i = pred["boxes"].cpu()
                    pred_scores_i = pred["scores"].cpu()
                    pred_labels_i = pred["labels"].cpu()

                tgt = targets[i]
                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()

                metric.update(
                    [
                        {
                            "boxes": pred_boxes_i,
                            "scores": pred_scores_i,
                            "labels": pred_labels_i,
                        }
                    ],
                    [{"boxes": gt_boxes, "labels": gt_labels}],
                )

                if len(gt_boxes) > 0 and len(pred_boxes_i) > 0:
                    ious = box_iou(pred_boxes_i, gt_boxes)
                    avg_iou = ious.max(dim=1).values.mean().item()
                else:
                    avg_iou = 0.0

                worst_heap.append(
                    (
                        avg_iou,
                        {
                            "image_idx": idx,
                            "avg_iou": avg_iou,
                            "image": images[i].cpu(),
                            "pred_boxes": pred_boxes_i,
                            "gt_boxes": gt_boxes,
                            "is_detr": is_detr,
                        },
                    )
                )

                if len(worst_heap) > 10:
                    worst_heap = heapq.nsmallest(10, worst_heap, key=lambda x: x[0])

        except Exception as e:
            print(f"[Eval ERROR] Skipping sample {idx}: {e}")
            continue
        finally:
            torch.cuda.empty_cache()

    final_scores = metric.compute()
    wandb.log(final_scores)
    wandb.summary.update(
        {
            "final_test_mAP": final_scores.get("map", -1),
            "final_test_mAP_50": final_scores.get("map_50", -1),
            "final_test_mAR_100": final_scores.get("mar_100", -1),
        }
    )

    return [item for _, item in sorted(worst_heap, key=lambda x: x[0])]


@torch.inference_mode()
def log_worst_predictions(results: List[Dict], count: int = 10):
    worst = sorted(results, key=lambda x: x["avg_iou"])[:count]

    for i, item in enumerate(worst):
        img = (item["image"] * 255).byte().clone()
        gt_boxes = item["gt_boxes"]
        pred_boxes = item["pred_boxes"]

        if item["is_detr"]:
            img_h, img_w = img.shape[1], img.shape[2]
            scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

            gt_boxes = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy") * scale

        gt_boxes = gt_boxes.clamp(min=0, max=max(img.shape[1], img.shape[2]))
        pred_boxes = pred_boxes.clamp(min=0, max=max(img.shape[1], img.shape[2]))

        img = draw_bounding_boxes(img, boxes=gt_boxes, colors="green", width=2)
        img = draw_bounding_boxes(img, boxes=pred_boxes, colors="red", width=2)

        wandb.log(
            {
                f"worst_pred_{i}": wandb.Image(
                    to_pil_image(img),
                    caption=f"IoU={item['avg_iou']:.2f} — RED: pred | GREEN: gt",
                )
            }
        )
