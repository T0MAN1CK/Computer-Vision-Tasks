import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from DETR.data.dataset import SKU110KDataset
from DETR.model.detr import DETR
from DETR.lightning_module import DETRLightningModule
from DETR.utils.train_utils import train_model

base_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = base_dir / "DETR" / "checkpoints"
checkpoint_name = "best-v1.ckpt"


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    pl.seed_everything(42)

    train_dataset = SKU110KDataset(
        csv_path=base_dir / "dataset/sample_SKU110K/annotations/annotations_train.csv",
        image_dir=base_dir / "dataset/sample_SKU110K/images",
        use_aug=True,
        resize_to=(800, 800),
    )
    val_dataset = SKU110KDataset(
        csv_path=base_dir / "dataset/sample_SKU110K/annotations/annotations_val.csv",
        image_dir=base_dir / "dataset/sample_SKU110K/images",
        use_aug=False,
        resize_to=(800, 800),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    model = DETR(
        num_classes=2,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    )

    lit_model = DETRLightningModule(model=model)

    train_model(
        model=lit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="DETR",
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name=checkpoint_name,
        max_epochs=5,
        check_val_every_n_epoch=1,
    )


if __name__ == "__main__":
    main()
