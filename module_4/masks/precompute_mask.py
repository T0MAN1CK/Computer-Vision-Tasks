import json
import torch
import numpy as np
from pathlib import Path
import cv2
from torchvision.io import read_image
import torchvision.transforms.functional as TF

# Parameters
CIRCLE_RADIUS = 10
CIRCLE_COLOR = 1.0
THICKNESS = -1


def generate_masks(image_dir, label_json, mask_dir, resize_to, original_size=None):
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(label_json, "r") as f:
        annotations = json.load(f)

    count = 0
    for entry in annotations.values():
        filename = entry["filename"]
        regions = entry.get("regions", {})

        img_path = image_dir / filename
        if not img_path.exists():
            continue

        img = read_image(str(img_path)).float() / 255.0
        img = TF.resize(img, resize_to)
        h, w = resize_to

        mask_np = np.zeros((h, w), dtype=np.float32)

        for region in regions.values():
            shape_attr = region.get("shape_attributes", {})
            if shape_attr.get("name") == "point":
                cx, cy = shape_attr.get("cx"), shape_attr.get("cy")
                if cx is not None and cy is not None:
                    if original_size:
                        orig_w, orig_h = original_size
                        cx = int(cx * (w / orig_w))
                        cy = int(cy * (h / orig_h))
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            mask_np,
                            (cx, cy),
                            radius=CIRCLE_RADIUS,
                            color=CIRCLE_COLOR,
                            thickness=THICKNESS,
                        )

        torch.save(
            torch.from_numpy(mask_np).unsqueeze(0),
            mask_dir / f"{Path(filename).stem}.pt",
        )
        count += 1

    print(f"Saved {count} masks to {mask_dir}")


if __name__ == "__main__":
    generate_masks(
        image_dir=Path("segmentation_dataset/train_data"),
        label_json=Path("segmentation_dataset/train_data/train_labels.json"),
        mask_dir=Path("segmentation_dataset/precomputed_masks"),
        resize_to=(256, 256),
    )

    generate_masks(
        image_dir=Path("segmentation_dataset/test_data"),
        label_json=Path("segmentation_dataset/test_data/test_labels_downscaled.json"),
        mask_dir=Path("segmentation_dataset/precomputed_masks_test"),
        resize_to=(768, 1024),
        original_size=(2560, 1920),
    )
