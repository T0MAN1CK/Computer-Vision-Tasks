import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
import wandb


def train_model(
    model_name: str,
    model,
    datamodule,
    config: dict,
    wandb_project: str,
    logs_dir: str,
    checkpoint_dir: str,
    group_name: str,
):
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=wandb_project,
        name=model_name,
        group=group_name,
        save_dir=logs_dir,
        log_model=False,
        config=config,
    )
    wandb_logger.watch(model, log="all", log_freq=100)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename=f"{model_name}-" + "{epoch:02d}-{val_acc:.2f}",
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

    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 50 + "\n")

    wandb.finish()
