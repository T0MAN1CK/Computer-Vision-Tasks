# ruff: noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.model import ViT
from shared.litmodule import LitClassifier
from shared.train_utils import train_model
from shared.datamodule import UniversalDataModule


def main():
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "Classification_data"
    logs_dir = base_dir / "ViT" / "wandblogs"
    checkpoint_dir = base_dir / "ViT" / "checkpoints"

    config = {
        "batch_size": 256,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "num_classes": 6,
        "max_epochs": 100,
        "architecture": "VisionTransformer",
    }

    backbone = ViT(n_classes=config["num_classes"])
    lit_model = LitClassifier(
        model=backbone,
        num_classes=config["num_classes"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    datamodule = UniversalDataModule(
        data_dir=str(data_dir),
        batch_size=config["batch_size"],
        val_split=0.2,
        resize=(144, 144),
        use_kornia_aug=True,
    )

    train_model(
        model_name="vit-bs256-lr3e4",
        model=lit_model,
        datamodule=datamodule,
        config=config,
        wandb_project="image-classification",
        logs_dir=str(logs_dir),
        checkpoint_dir=str(checkpoint_dir),
        group_name="ViT",
    )


if __name__ == "__main__":
    main()
