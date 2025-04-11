# ruff: noqa: E402

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from models.model import FinetunedResNet
from shared.litmodule import LitClassifier
from shared.train_utils import train_model
from shared.datamodule import UniversalDataModule


def main():
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "Classification_data"
    logs_dir = base_dir / "finetuned_cnn" / "wandblogs"
    checkpoint_dir = base_dir / "finetuned_cnn" / "checkpoints"

    config = {
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_classes": 6,
        "max_epochs": 25,
        "architecture": "finetuned-resnet18",
    }

    backbone = FinetunedResNet(num_classes=config["num_classes"])
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
        resize=(150, 150),
        use_kornia_aug=True,
    )

    train_model(
        model_name="resnet18-bs256-lr1e3",
        model=lit_model,
        datamodule=datamodule,
        config=config,
        wandb_project="image-classification",
        logs_dir=str(logs_dir),
        checkpoint_dir=str(checkpoint_dir),
        group_name="FinetunedCNN",
    )


if __name__ == "__main__":
    main()
