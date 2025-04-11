import torch
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision
from typing import List, Dict
from tqdm import tqdm
import heapq


@torch.inference_mode()
def evaluate_map_metrics(model: torch.nn.Module, dataloader) -> Dict:
    model.eval()
    metric = MeanAveragePrecision()
    worst_heap = []  # stores (avg_iou, result_dict)

    for idx, (image, target) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating"
    ):
        try:
            image = image.cuda()
            preds = model.model([image.squeeze(0)])
            pred = preds[0]

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()

            gt_boxes = target["boxes"][0].cpu()
            gt_labels = target["labels"][0].cpu()

            metric.update(
                [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
                [{"boxes": gt_boxes, "labels": gt_labels}],
            )

            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                avg_iou = 0.0
            else:
                ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                avg_iou = ious.max(dim=1).values.mean().item()

            # Store only top 10 worst
            item = {
                "image_idx": idx,
                "avg_iou": avg_iou,
                "image": image.cpu().squeeze(0),
                "pred_boxes": pred_boxes,
                "gt_boxes": gt_boxes,
            }
            heapq.heappush(worst_heap, (avg_iou, item))
            if len(worst_heap) > 10:
                heapq.heappop(worst_heap)

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Eval ERROR] Skipping sample {idx}: {e}")
            continue

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
        img = (item["image"] * 255).byte()
        drawn = draw_bounding_boxes(
            img, boxes=item["gt_boxes"], colors="green", width=2
        )
        drawn = draw_bounding_boxes(
            drawn, boxes=item["pred_boxes"], colors="red", width=2
        )
        wandb.log(
            {
                f"worst_pred_{i}": wandb.Image(
                    to_pil_image(drawn), caption=f"IoU={item['avg_iou']:.2f}"
                )
            }
        )
