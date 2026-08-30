import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_image
import pandas as pd
import kornia.augmentation as K
import warnings

warnings.filterwarnings(
    "ignore", message="Default grid_sample and affine_grid behavior has changed"
)


class RodSegmentationDataset(Dataset):
    def __init__(self, csv_path, fold=None, train=True, apply_aug=True):
        self.df = pd.read_csv(csv_path)
        self.train = train
        self.apply_aug = apply_aug

        if fold is not None:
            if train:
                self.df = self.df[self.df["fold"] != fold]
            else:
                self.df = self.df[self.df["fold"] == fold]

        self.image_paths = self.df["image_path"].tolist()
        self.mask_paths = self.df["mask_path"].tolist()

        self.spatial_aug = None
        if self.apply_aug:
            self.spatial_aug = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.3),
                K.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), p=0.4
                ),
                data_keys=["input", "mask"],
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx]).float() / 255.0  # (3, H, W)
        mask = torch.load(self.mask_paths[idx])  # (1, H, W)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        if self.spatial_aug:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            image_aug, mask_aug = self.spatial_aug(image, mask)
            image = image_aug.squeeze(0)
            mask = mask_aug.squeeze(0).clamp(0, 1)

        image = F.interpolate(
            image.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
        )[0]
        mask = F.interpolate(
            mask.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=True
        )[0]

        return image, mask
