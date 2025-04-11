import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def train_model(
    model,
    train_loader,
    val_loader,
    model_name: str,
    checkpoint_dir: str = "FasterRCNN/checkpoints",
    project_name: str = "module_3",
    max_epochs: int = 10,
    check_val_every_n_epoch: int = 2,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Wandb Logger ---
    wandb_logger = WandbLogger(project=project_name, name=model_name, log_model=False)

    # --- Checkpoint Callback ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # --- EarlyStopping Callback ---
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=2, verbose=True, mode="min"
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 50)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Validation loss: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 50 + "\n")
