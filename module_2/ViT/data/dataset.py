from PIL import Image
import os
from typing import List
from torch.utils.data import Dataset

CLASS_NAMES = ["building", "forest", "glacier", "mountain", "sea", "street"]


class CustomImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        kornia_aug=None,
        training=False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.kornia_aug = kornia_aug
        self.training = training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.training and self.kornia_aug:
            image = self.kornia_aug(image.unsqueeze(0)).squeeze(0)

        return image, label


def list_image_paths_and_labels(split):
    root = os.path.join("module_2", "Classification_data", split)
    samples = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(("jpg", "jpeg", "png")):
                samples.append((os.path.join(class_dir, fname), i))
    return samples
