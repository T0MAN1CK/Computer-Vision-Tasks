import json
import pandas as pd
from pathlib import Path

# Paths
image_dir = Path("segmentation_dataset/train_data")
mask_dir = Path("segmentation_dataset/precomputed_masks")
label_json = image_dir / "train_labels.json"

# Load annotations
with open(label_json, "r") as f:
    annotations = json.load(f)

records = []
for entry in annotations.values():
    filename = entry["filename"]
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    regions = entry.get("regions", {})
    is_empty = int(len(regions) == 0)

    records.append(
        {
            "filename": filename,
            "image_path": str(image_dir / filename),
            "mask_path": str(mask_dir / f"{Path(filename).stem}.pt"),
            "empty": is_empty,
        }
    )

df = pd.DataFrame(records)
df.to_csv("segmentation_dataset/train.csv", index=False)

print("train.csv saved to segmentation_dataset/train.csv")
