import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from typing import Type, Optional
from shared.datasets import GeneralImageDataset


class UniversalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_cls: Type[torch.utils.data.Dataset] = GeneralImageDataset,
        batch_size: int = 64,
        val_split: float = 0.2,
        num_workers: int = 4,
        resize: Optional[tuple] = (150, 150),
        normalize: bool = True,
        use_kornia_aug: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.resize = resize
        self.normalize = normalize
        self.use_kornia_aug = use_kornia_aug

    def setup(self, stage: Optional[str] = None):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Normalize(mean, std) if self.normalize else None

        if stage == "fit" or stage is None:
            full_dataset = self.dataset_cls(
                data_dir=os.path.join(self.data_dir, "train"),
                transform=transform,
                augment=True,
                resize=self.resize,
                use_kornia=self.use_kornia_aug,
            )
            train_size = int((1 - self.val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.train_dataset.dataset.augment = True
            self.val_dataset.dataset.augment = False

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(
                data_dir=os.path.join(self.data_dir, "test"),
                transform=transform,
                augment=False,
                resize=self.resize,
                use_kornia=self.use_kornia_aug,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
