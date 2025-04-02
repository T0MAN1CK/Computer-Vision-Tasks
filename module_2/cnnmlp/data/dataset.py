import os
import cv2
import torch
import kornia.augmentation as K
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def read_image(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    assert img is not None, f"Image at {path} could not be read."
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_tensor = torch.from_numpy(img).float() / 255.0  # Normalize to [0, 1]
    return img_tensor.permute(2, 0, 1)  # HWC -> CHW


class ClassificationDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        kornia_aug: Optional[nn.Module] = None,
        training: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.kornia_aug = kornia_aug
        self.training = training

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = read_image(img_path)
        if self.training and self.kornia_aug:
            img = self.kornia_aug(img.unsqueeze(0)).squeeze(0)
        return img, label


class ClassificationDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.kornia_aug = nn.Sequential(
            K.RandomRotation(degrees=25.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        )

    def setup(self, stage: str = None):
        train_paths, train_labels = self._load_split("train")
        test_paths, test_labels = self._load_split("test")

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths,
            train_labels,
            test_size=0.1,
            stratify=train_labels,
            random_state=42,
        )

        self.train_dataset = ClassificationDataset(
            train_paths, train_labels, kornia_aug=self.kornia_aug, training=True
        )
        self.val_dataset = ClassificationDataset(val_paths, val_labels, training=False)
        self.test_dataset = ClassificationDataset(
            test_paths, test_labels, training=False
        )

    def _load_split(self, split: str) -> Tuple[List[str], List[int]]:
        split_path = os.path.normpath(os.path.join(self.data_dir, split))
        all_image_paths = []
        all_labels = []
        for class_name in CLASS_NAMES:
            class_path = os.path.normpath(os.path.join(split_path, class_name))
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(class_path, fname)
                    all_image_paths.append(fpath)
                    all_labels.append(CLASS_TO_IDX[class_name])
        return all_image_paths, all_labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )
