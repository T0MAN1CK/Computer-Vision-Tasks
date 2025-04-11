import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from FasterRCNN.data.dataset import SKU110KDataset
from FasterRCNN.lightning_module import FasterRCNNLightningModule
from FasterRCNN.utils.train_utils import train_model


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    pl.seed_everything(42)

    base_dir = Path(__file__).resolve().parent.parent
    checkpoint_dir = base_dir / "FasterRCNN" / "checkpoints"

    train_dataset = SKU110KDataset(
        csv_path=base_dir / "dataset/sample_SKU110K/annotations/annotations_train.csv",
        image_dir=base_dir / "dataset/sample_SKU110K/images",
        use_aug=True,
    )
    val_dataset = SKU110KDataset(
        csv_path=base_dir / "dataset/sample_SKU110K/annotations/annotations_val.csv",
        image_dir=base_dir / "dataset/sample_SKU110K/images",
        use_aug=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    model = FasterRCNNLightningModule(num_classes=2)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="FasterRCNN_TorchVision",
        checkpoint_dir=str(checkpoint_dir),
        max_epochs=5,
        check_val_every_n_epoch=1,
    )


if __name__ == "__main__":
    main()
