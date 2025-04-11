import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics


class FinetunedResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 6,
        lr: float = 1e-3,
        finetune: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

        #  classifier with dropout and regularization
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "targets": y}

    def configure_optimizers(self):
        # AdamW with weight decay for regularization
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,  # L2 regularization
        )
