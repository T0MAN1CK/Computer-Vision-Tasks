import json
from pathlib import Path

# Paths
TEST_DIR = Path("segmentation_dataset/test_data")
LABELS_PATH = TEST_DIR / "test_labels.json"
OUTPUT_PATH = TEST_DIR / "test_labels_downscaled.json"

# Original and target resolutions
orig_w, orig_h = 4160, 3120
target_w, target_h = 1024, 768

# Load original labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


def rescale(val, orig, target):
    return round(val * target / orig)


rescaled_labels = {}
for key, item in labels.items():
    new_item = {"filename": item["filename"], "regions": {}}

    for region_id, region in item["regions"].items():
        cx = region["shape_attributes"]["cx"]
        cy = region["shape_attributes"]["cy"]

        new_cx = rescale(cx, orig_w, target_w)
        new_cy = rescale(cy, orig_h, target_h)

        new_item["regions"][region_id] = {
            "shape_attributes": {"cx": new_cx, "cy": new_cy}
        }

    rescaled_labels[key] = new_item

with open(OUTPUT_PATH, "w") as f:
    json.dump(rescaled_labels, f, indent=2)

print(f"Rescaled labels saved to {OUTPUT_PATH}")
