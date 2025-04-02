import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader
from finetuned_cnn.data.dataset import ImageClassificationDataset
from torchvision import transforms
from torch.utils.data import random_split


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage: str):
        common_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if stage == "fit" or stage is None:
            full_train_dataset = ImageClassificationDataset(
                data_dir=os.path.join(self.data_dir, "train"),
                transform=common_transform,
                augment=True,
            )

            train_size = int((1 - self.val_split) * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            self.train_dataset.dataset.augment = True
            self.val_dataset.dataset.augment = False

        if stage == "test" or stage is None:
            self.test_dataset = ImageClassificationDataset(
                data_dir=os.path.join(self.data_dir, "test"),
                transform=common_transform,
                augment=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
