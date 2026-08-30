import torch
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss_dict.values())
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        val_loss = sum(loss_dict.values())
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
