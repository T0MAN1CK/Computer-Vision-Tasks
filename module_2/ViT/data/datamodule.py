import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import kornia.augmentation as K
from kornia.constants import Resample
import torch

from data.dataset import CustomImageDataset, list_image_paths_and_labels


class KorniaAugmentation(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=10.0, resample=Resample.BILINEAR),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            K.RandomAffine(degrees=10, translate=(0.1, 0.1), p=0.3),
        )


class ImageClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        all_samples = list_image_paths_and_labels("train")

        train_samples, val_samples = train_test_split(
            all_samples,
            test_size=0.2,
            stratify=[label for _, label in all_samples],
            random_state=42,
        )

        transform = transforms.Compose(
            [
                transforms.Resize((144, 144)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        if stage == "fit" or stage is None:
            train_paths, train_labels = zip(*train_samples)
            val_paths, val_labels = zip(*val_samples)

            self.train_dataset = CustomImageDataset(
                list(train_paths),
                list(train_labels),
                transform=transform,
                kornia_aug=KorniaAugmentation(),
                training=True,
            )

            self.val_dataset = CustomImageDataset(
                list(val_paths), list(val_labels), transform=transform
            )

        if stage == "test" or stage is None:
            test_samples = list_image_paths_and_labels("test")
            test_paths, test_labels = zip(*test_samples)

            self.test_dataset = CustomImageDataset(
                list(test_paths), list(test_labels), transform=transform
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
