import torch
from pathlib import Path
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from models.cnn_mlp import CNNMLPClassifier
from data.dataset import ClassificationDataModule


def main():
    pl.seed_everything(42)

    current_dir = Path(__file__).parent.absolute()
    data_dir = current_dir.parent / "Classification_data"
    checkpoint_dir = current_dir / "trained_models"
    logs_dir = current_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_classes": 6,
        "max_epochs": 25,
        "architecture": "CNN+MLP",
    }

    data_module = ClassificationDataModule(
        data_dir=str(data_dir),
        batch_size=config["batch_size"],
    )

    model = CNNMLPClassifier(
        num_classes=config["num_classes"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        dirpath=str(checkpoint_dir),
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
        project="cnnmlp-classification",
        name="training-run",
        save_dir=str(logs_dir),
        log_model=False,
        config=config,
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=50,
    )

    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model, datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
