import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from pathlib import Path
import cv2
import numpy as np
import kornia.augmentation as K
from typing import Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SKU110KDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        use_aug=False,
        visualize=False,
        resize_to: Optional[Tuple[int, int]] = None,
        model_type: str = "detr",  # "detr" or "fasterrcnn"
    ):
        assert model_type in ("detr", "fasterrcnn")
        self.model_type = model_type
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
        self.resize_to = resize_to

        self.image_groups = self.df.groupby("image_name")
        self.image_names = [
            n for n in self.image_groups.groups if (self.image_dir / n).exists()
        ]

        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            K.RandomPerspective(p=0.3),
            K.ColorJitter(0.3, 0.3, 0.3, 0.1),
            data_keys=["input", "bbox"],
        ).to("cpu")

    def __len__(self):
        return len(self.image_names)

    def xyxy_to_polygon(self, boxes):
        polygons = []
        for box in boxes:
            x1, y1, x2, y2 = box
            polygons.append(torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
        return torch.stack(polygons)

    def polygon_to_xyxy(self, polygons):
        x = polygons[:, :, 0]
        y = polygons[:, :, 1]
        return torch.stack(
            [x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values], dim=1
        )

    def box_xyxy_to_cxcywh(self, boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes.unbind(1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack((cx, cy, w, h), dim=1)

    def __getitem__(self, idx: int, attempt: int = 0):
        try:
            image_name = self.image_names[idx]
            rows = self.image_groups.get_group(image_name)
            image_path = self.image_dir / image_name

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"[Dataset] Failed to read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            boxes = torch.tensor(
                rows[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
            )
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            image_tensor = F.to_tensor(image)

            if self.use_aug:
                image_tensor = image_tensor.unsqueeze(0)
                polygons = self.xyxy_to_polygon(boxes).unsqueeze(0)
                image_tensor, polygons = self.aug(image_tensor, polygons)
                boxes = self.polygon_to_xyxy(polygons.squeeze(0)).to("cpu")
                image_tensor = image_tensor.squeeze(0).to("cpu")

                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                valid = (box_w > 1) & (box_h > 1)
                boxes = boxes[valid]
                labels = labels[valid]

            if self.model_type == "detr" and self.resize_to is not None:
                new_h, new_w = self.resize_to
                orig_h, orig_w = image_tensor.shape[1:]

                scale = min(new_w / orig_w, new_h / orig_h)
                resized_h = int(orig_h * scale)
                resized_w = int(orig_w * scale)

                image_tensor = torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                # Pad image to final size
                padded = torch.zeros((3, new_h, new_w), dtype=image_tensor.dtype)
                padded[:, :resized_h, :resized_w] = image_tensor
                image_tensor = padded

                # Scale boxes
                boxes[:, [0, 2]] *= scale
                boxes[:, [1, 3]] *= scale

                # Normalize boxes to [0, 1] in cxcywh format
                boxes = self.box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([new_w, new_h, new_w, new_h])
                boxes = boxes.clamp(0, 1)

            elif self.model_type == "fasterrcnn":
                boxes = boxes.clamp(0)

            target = {
                "boxes": boxes
                if boxes.numel() > 0
                else torch.zeros((0, 4), dtype=torch.float32),
                "labels": labels
                if labels.numel() > 0
                else torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor(idx),
            }

            if self.visualize:
                img_np = image_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                from shared.visualize import draw_boxes_on_image

                img_with_boxes = draw_boxes_on_image(img_np, boxes)
                cv2.imwrite(
                    f"augmented_{idx}.jpg",
                    cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR),
                )

            return image_tensor, target

        except Exception as e:
            print(f"[Dataset ERROR] Skipping index {idx}: {e}")
            if attempt > 5:
                raise RuntimeError(f"Too many failures at index {idx}")
            return self.__getitem__((idx + 1) % len(self), attempt + 1)
