import torch
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision
from typing import List, Dict
from torchvision.ops import box_convert
import tqdm


@torch.inference_mode()
def evaluate_map_metrics(model: torch.nn.Module, dataloader) -> Dict:
    model.eval()
    metric = MeanAveragePrecision()
    results = []

    for idx, (images, targets) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating"
    ):
        try:
            images = [img.cuda() for img in images]

            if isinstance(targets[0], dict):
                targets = [
                    {
                        k: v.cuda() if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]
            else:
                raise ValueError(
                    f"Expected targets to be list of dicts, but got: {targets}"
                )

            outputs = model([images[0]])
            logits = outputs["pred_logits"][0]
            boxes = outputs["pred_boxes"][0]

            probs = logits.softmax(-1)
            scores, labels = probs[..., :-1].max(-1)  # exclude "no-object" class

            keep = scores > 0.3
            pred_boxes = boxes[keep].cpu()
            pred_scores = scores[keep].cpu()
            pred_labels = labels[keep].cpu()

            gt_boxes = targets[0]["boxes"].cpu()
            gt_labels = targets[0]["labels"].cpu()

            preds = [
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                }
            ]
            targets_fmt = [
                {
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                }
            ]
            metric.update(preds, targets_fmt)

            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                avg_iou = 0.0
            else:
                ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                avg_iou = ious.max(dim=1).values.mean().item()

            results.append(
                {
                    "image_idx": idx,
                    "avg_iou": avg_iou,
                    "image": images[0].cpu(),
                    "pred_boxes": pred_boxes,
                    "gt_boxes": gt_boxes,
                }
            )

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
    return results


@torch.inference_mode()
def log_worst_predictions(results: List[Dict], count: int = 10):
    worst = sorted(results, key=lambda x: x["avg_iou"])[:count]

    for i, item in enumerate(worst):
        img = (item["image"] * 255).byte()

        gt_boxes_xyxy = box_convert(item["gt_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
        drawn = draw_bounding_boxes(img, boxes=gt_boxes_xyxy, colors="green", width=2)

        pred_boxes_xyxy = box_convert(
            item["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy"
        )
        drawn = draw_bounding_boxes(drawn, boxes=pred_boxes_xyxy, colors="red", width=2)
        wandb.log(
            {
                f"worst_pred_{i}": wandb.Image(
                    to_pil_image(drawn), caption=f"IoU={item['avg_iou']:.2f}"
                )
            }
        )
