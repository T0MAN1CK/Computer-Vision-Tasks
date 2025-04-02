import os
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
import kornia.augmentation as K


CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        super().__init__()
        self.samples = []
        self.transform = transform
        self.augment = augment
        self.kornia_aug = (
            torch.nn.Sequential(
                K.RandomRotation(degrees=15.0),
                K.RandomHorizontalFlip(p=0.3),
                K.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                K.RandomPerspective(distortion_scale=0.2, p=0.3),
            )
            if augment
            else None
        )

        for class_name in CLASS_NAMES:
            class_path = os.path.join(data_dir, class_name)
            image_paths = glob(os.path.join(class_path, "*.jpg"))
            self.samples.extend([(p, CLASS_TO_IDX[class_name]) for p in image_paths])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        image = cv2.imread(img_path)
        assert image is not None, f"Failed to load image: {img_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Apply augmentations
        if self.augment and self.kornia_aug:
            image = self.kornia_aug(image.unsqueeze(0)).squeeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label
