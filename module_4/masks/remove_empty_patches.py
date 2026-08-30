from pathlib import Path

# Paths
image_dir = Path("segmentation_dataset/train_data")
mask_dir = Path("segmentation_dataset/precomputed_masks")

# Suffix patterns to remove
suffixes_to_remove = ["_12", "_13", "_14", "_15"]

deleted_images = 0
deleted_masks = 0

for img_file in image_dir.glob("*"):
    stem = img_file.stem
    if any(stem.endswith(suffix) for suffix in suffixes_to_remove):
        img_file.unlink()
        deleted_images += 1

        mask_file = mask_dir / f"{stem}.pt"
        if mask_file.exists():
            mask_file.unlink()
            deleted_masks += 1

print(f"Deleted {deleted_images} images and {deleted_masks} masks")
