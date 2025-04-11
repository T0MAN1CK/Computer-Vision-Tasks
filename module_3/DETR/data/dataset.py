import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from pathlib import Path
import cv2
import numpy as np
import kornia.augmentation as K
from typing import Optional, Tuple


class SKU110KDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        use_aug=False,
        visualize=False,
        resize_to: Optional[Tuple[int, int]] = None,
    ):
        self.df = pd.read_csv(csv_path, header=None)
        self.df.columns = [
            "image_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "class",
            "image_width",
            "image_height",
        ]

        self.image_dir = Path(image_dir)
        self.use_aug = use_aug
        self.visualize = visualize

        self.image_groups = self.df.groupby("image_name")
        self.image_names = list(self.image_groups.groups.keys())

        self.resize_to = resize_to

        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            K.RandomPerspective(p=0.3),
            K.ColorJitter(0.3, 0.3, 0.3, 0.1),
            data_keys=["input", "bbox"],
        ).to("cpu")

        valid_names = [
            name for name in self.image_names if (self.image_dir / name).exists()
        ]
        self.image_names = valid_names

    def __len__(self):
        return len(self.image_names)

    def xyxy_to_polygon(self, boxes):
        """Convert (x1, y1, x2, y2) -> 4-point polygon for each box"""
        polygons = []
        for box in boxes:
            x1, y1, x2, y2 = box
            poly = torch.tensor(
                [
                    [x1, y1],  # top-left
                    [x2, y1],  # top-right
                    [x2, y2],  # bottom-right
                    [x1, y2],  # bottom-left
                ]
            )
            polygons.append(poly)
        return torch.stack(polygons)  # (N, 4, 2)

    def polygon_to_xyxy(self, polygons):
        """Convert 4-point polygon -> (x1, y1, x2, y2) bounding boxes"""
        x_coords = polygons[:, :, 0]
        y_coords = polygons[:, :, 1]
        x1 = x_coords.min(dim=1).values
        y1 = y_coords.min(dim=1).values
        x2 = x_coords.max(dim=1).values
        y2 = y_coords.max(dim=1).values
        return torch.stack([x1, y1, x2, y2], dim=1)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        rows = self.image_groups.get_group(image_name)

        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"[Dataset] Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(
                f"[Dataset] Failed to read image with OpenCV: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = rows[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        boxes = torch.tensor(boxes)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image_tensor = F.to_tensor(image)  # (C, H, W)

        if self.use_aug:
            image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
            polygons = self.xyxy_to_polygon(boxes).unsqueeze(0)  # (1, N, 4, 2)

            image_tensor, polygons = self.aug(image_tensor, polygons)

            boxes = self.polygon_to_xyxy(polygons.squeeze(0)).to("cpu")
            image_tensor = image_tensor.squeeze(0).to("cpu")

            #  Clamp boxes to image size
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

            #  Filter out invalid boxes
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid = (box_w > 1) & (box_h > 1)
            boxes = boxes[valid]
            labels = labels[valid]

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(idx)}

        if self.visualize:
            from DETR.utils.vis_utils import draw_boxes_on_image

            img_np = image_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_with_boxes = draw_boxes_on_image(img_np, boxes)
            cv2.imwrite(
                f"augmented_{idx}.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            )

        if self.resize_to is not None:
            orig_h, orig_w = image_tensor.shape[-2:]
            image_tensor = F.resize(image_tensor, self.resize_to)

            new_h, new_w = self.resize_to
            scale_w = new_w / orig_w
            scale_h = new_h / orig_h

            boxes = target["boxes"]
            boxes = boxes * torch.tensor([scale_w, scale_h, scale_w, scale_h])
            target["boxes"] = boxes

        return image_tensor, target
