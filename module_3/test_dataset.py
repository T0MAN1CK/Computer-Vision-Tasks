import random
from pathlib import Path
import os
import cv2
import numpy as np

from shared.dataset import SKU110KDataset as DETRDataset
from shared.dataset import SKU110KDataset as FRCNNDataset
from shared.visualize import draw_boxes_on_image

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


def visualize_dataset(name: str, dataset_cls):
    root = Path(__file__).resolve().parent / "dataset/sample_SKU110K"
    image_dir = root / "images"
    ann_path = root / "annotations" / "annotations_train.csv"

    dataset_orig = dataset_cls(str(ann_path), str(image_dir), use_aug=False)
    dataset_aug = dataset_cls(str(ann_path), str(image_dir), use_aug=True)

    idx = random.randint(0, len(dataset_orig) - 1)
    print(f"[{name}] Visualizing index {idx}...")

    img_orig, target_orig = dataset_orig[idx]
    img_aug, target_aug = dataset_aug[idx]

    boxes_orig = target_orig["boxes"]
    boxes_aug = target_aug["boxes"]

    np_orig = (img_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    np_aug = (img_aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    img_with_boxes_orig = draw_boxes_on_image(np_orig, boxes_orig)
    img_with_boxes_aug = draw_boxes_on_image(np_aug, boxes_aug)

    combined = np.hstack((img_with_boxes_orig, img_with_boxes_aug))
    save_path = Path(__file__).resolve().parent / f"{name}_compare_{idx}.jpg"
    cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[{name}] Saved comparison to {save_path}")


if __name__ == "__main__":
    visualize_dataset("FasterRCNN", FRCNNDataset)
    visualize_dataset("DETR", DETRDataset)
