import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.model import ViTModule
from data.datamodule import ImageClassificationDataModule
from pytorch_lightning.callbacks import EarlyStopping


def main():
    pl.seed_everything(42)

    # Config
    data_dir = "module_2/Classification_data"
    logs_dir = "module_2/ViT/wandblogs"
    checkpoint_dir = "module_2/ViT/checkpoints"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = {
        "batch_size": 256,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "num_classes": 6,
        "max_epochs": 100,
        "architecture": "VisionTransformer",
    }

    wandb_logger = WandbLogger(
        project="vit-classification",
        name="training-run",
        save_dir=logs_dir,
        log_model=False,
        config=config,
    )

    model = ViTModule(
        num_classes=config["num_classes"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    data_module = ImageClassificationDataModule(
        data_dir=data_dir, batch_size=config["batch_size"]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="vit-{epoch:02d}-{val_acc:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    early_stop = EarlyStopping(monitor="val_acc", patience=10, mode="max", verbose=True)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        log_every_n_steps=50,
    )

    wandb_logger.watch(model, log="all", log_freq=100)

    trainer.fit(model, datamodule=data_module)

    print("\n" + "=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 50 + "\n")

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
