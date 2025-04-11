import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


def train_model(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    checkpoint_dir: str = "FasterRCNN/checkpoints",
    checkpoint_name: str = "best.ckpt",
    project_name: str = "module_3",
    max_epochs: int = 10,
    check_val_every_n_epoch: int = 5,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Wandb Logger ---
    wandb_logger = WandbLogger(project=project_name, name=model_name, log_model=False)

    # --- Checkpoint Callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_name.replace(".ckpt", ""),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 50 + "\n")
