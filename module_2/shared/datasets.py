# module_2/shared/datasets.py

import os
import cv2
import torch
from glob import glob
from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
import kornia.augmentation as K
from shared.constants import CLASS_TO_IDX


class GeneralImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
        resize: Tuple[int, int] = (150, 150),
        use_kornia: bool = True,
    ):
        self.samples = []
        self.transform = transform
        self.augment = augment
        self.resize = resize

        if use_kornia:
            self.kornia_aug = torch.nn.Sequential(
                K.RandomRotation(degrees=15.0),
                K.RandomHorizontalFlip(p=0.3),
                K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                K.RandomPerspective(distortion_scale=0.2, p=0.3),
            )
        else:
            self.kornia_aug = None

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_path = os.path.join(data_dir, class_name)
            image_paths = glob(
                os.path.join(class_path, "*.[jp][pn]g")
            )  # handles jpg/jpeg/png
            self.samples.extend([(p, class_idx) for p in image_paths])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]

        image = cv2.imread(img_path)
        assert image is not None, f"Failed to load image: {img_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.augment and self.kornia_aug:
            image = self.kornia_aug(image.unsqueeze(0)).squeeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label
