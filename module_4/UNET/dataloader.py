import torch
from torch.utils.data import DataLoader
from UNET.dataset import RodSegmentationDataset


def prepare_loaders(csv_path, fold, batch_size=32, num_workers=4, debug=False):
    if debug:
        print("DEBUG MODE: Only loading 80 train + 40 val samples")

    train_dataset = RodSegmentationDataset(
        csv_path=csv_path, fold=fold, train=True, apply_aug=True
    )

    val_dataset = RodSegmentationDataset(
        csv_path=csv_path, fold=fold, train=False, apply_aug=False
    )

    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(80)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(40)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
