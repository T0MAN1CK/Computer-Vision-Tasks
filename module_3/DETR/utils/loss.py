import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from torchvision.ops import box_convert


class HungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes):
        bs, num_queries = pred_logits.shape[:2]

        # Flatten predictions
        out_prob = pred_logits.softmax(-1)
        out_bbox = pred_boxes

        indices = []
        for b in range(bs):
            cost_class = -out_prob[b][:, tgt_labels[b]]

            pred_boxes_xyxy = box_convert(out_bbox[b], in_fmt="cxcywh", out_fmt="xyxy")
            tgt_boxes_xyxy = box_convert(tgt_boxes[b], in_fmt="cxcywh", out_fmt="xyxy")

            cost_bbox = torch.cdist(out_bbox[b], tgt_boxes[b], p=1)

            cost_giou = -generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)

            C = cost_bbox + cost_giou + cost_class
            C = C.cpu()
            i, j = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
            )
        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, **kwargs):
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip([t["labels"] for t in targets], indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, **kwargs):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        src_boxes_xyxy = box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        target_boxes_xyxy = box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        loss_bbox = F.l1_loss(
            src_boxes, target_boxes, reduction="none"
        ).sum() / src_boxes.size(0)
        loss_giou = (
            1
            - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)).mean()
        )

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        valid_targets = []
        valid_outputs_logits = []
        valid_outputs_boxes = []

        for logit, box, tgt in zip(
            outputs["pred_logits"], outputs["pred_boxes"], targets
        ):
            if tgt["boxes"].numel() > 0:
                valid_outputs_logits.append(logit.unsqueeze(0))
                valid_outputs_boxes.append(box.unsqueeze(0))
                valid_targets.append(tgt)

        if not valid_targets:
            return {
                "total_loss": torch.tensor(0.0, device=outputs["pred_logits"].device)
            }

        outputs_logits = torch.cat(valid_outputs_logits, dim=0)
        outputs_boxes = torch.cat(valid_outputs_boxes, dim=0)

        indices = self.matcher(
            outputs_logits,
            outputs_boxes,
            [t["labels"] for t in valid_targets],
            [t["boxes"] for t in valid_targets],
        )

        patched_outputs = {
            "pred_logits": outputs_logits,
            "pred_boxes": outputs_boxes,
        }

        losses = {}
        for loss in self.losses:
            losses.update(
                getattr(self, f"loss_{loss}")(patched_outputs, valid_targets, indices)
            )

        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses)
        losses["total_loss"] = total_loss
        return losses
