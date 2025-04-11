import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from DETR.utils.loss import HungarianMatcher, SetCriterion


def loss_fn(pred_logits, pred_boxes, tgt_labels, tgt_boxes):
    # Flatten predictions: [batch_size, num_queries, num_classes+1]
    bs, num_queries, num_classes_plus1 = pred_logits.shape
    pred_logits = pred_logits.reshape(bs * num_queries, num_classes_plus1)
    # Repeat ground truth to align with predicted queries (TEMP workaround for mismatch)
    tgt_labels = tgt_labels[: bs * num_queries]  # clip or pad as eeded
    cls_loss = nn.CrossEntropyLoss()(pred_logits, tgt_labels)
    pred_boxes = pred_boxes.reshape(bs * num_queries, 4)
    tgt_boxes = tgt_boxes[: bs * num_queries]  # clip or pad
    bbox_loss = nn.L1Loss()(pred_boxes, tgt_boxes)
    return cls_loss + bbox_loss


class DETRLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        matcher = HungarianMatcher()
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "boxes"],
        )

    def forward(self, images: list[torch.Tensor]) -> dict:
        images = torch.stack(images)  # (B, C, H, W)
        pred_logits, pred_boxes = self.model(images)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    def step(self, batch, stage: str):
        images, targets = batch
        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict["total_loss"]
        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
