import json
from pathlib import Path

# Paths
IMAGE_DIR = Path("segmentation_dataset/train_data")
LABEL_JSON = IMAGE_DIR / "train_labels.json"

# Load annotations
with open(LABEL_JSON, "r") as f:
    annotations = json.load(f)

# Remove entries for missing images
cleaned_annotations = {}
for key, entry in annotations.items():
    image_path = IMAGE_DIR / entry["filename"]
    if image_path.exists():
        cleaned_annotations[key] = entry

# Save cleaned JSON
with open(LABEL_JSON, "w") as f:
    json.dump(cleaned_annotations, f, indent=2)

print(f"Cleaned train_labels.json. Remaining images: {len(cleaned_annotations)}")
