# ruff: noqa: E402

import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
module_2_dir = os.path.dirname(script_dir)
sys.path.insert(0, module_2_dir)

from finetuned_cnn.models.model import FinetunedResNet
from finetuned_cnn.datamodule import ClassificationDataModule


def main():
    pl.seed_everything(42)

    os.makedirs("module_2/finetuned_cnn/checkpoints", exist_ok=True)
    os.makedirs("module_2/finetuned_cnn/wandblogs", exist_ok=True)

    model = FinetunedResNet(num_classes=6, lr=1e-3, finetune=True)
    datamodule = ClassificationDataModule(
        data_dir="module_2/Classification_data", batch_size=256, val_split=0.2
    )

    wandb_logger = WandbLogger(
        project="finetuned-cnn-classification",
        name="training-run",
        save_dir="module_2/finetuned_cnn/wandblogs",
        log_model=False,
        config={
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "num_classes": 6,
            "max_epochs": 25,
            "architecture": "finetuned-resnet18",
        },
    )
    wandb_logger.watch(model, log="all", log_freq=100)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="module_2/finetuned_cnn/checkpoints",
        filename="resnet18-{epoch:02d}-{val_acc:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        precision=16,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 50)
    print(
        f"Best model achieved validation accuracy: {checkpoint_callback.best_model_score:.4f}"
    )
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("=" * 50 + "\n")

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
