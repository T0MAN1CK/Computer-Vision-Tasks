import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from UNET.dataset import RodSegmentationDataset
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

output_dir = Path("debug_outputs_augmented")
output_dir.mkdir(exist_ok=True)

dataset_raw = RodSegmentationDataset(
    csv_path="segmentation_dataset/train_folds.csv", fold=0, train=True, apply_aug=False
)

dataset_aug = RodSegmentationDataset(
    csv_path="segmentation_dataset/train_folds.csv", fold=0, train=True, apply_aug=True
)

n_samples = 5
for idx in range(n_samples):
    image_raw, mask_raw = dataset_raw[idx]
    image_aug, mask_aug = dataset_aug[idx]

    image_raw_np = image_raw.permute(1, 2, 0).numpy()
    mask_raw_np = mask_raw.squeeze().numpy()
    image_aug_np = image_aug.permute(1, 2, 0).numpy()
    mask_aug_np = mask_aug.squeeze().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(
        (image_raw_np - image_raw_np.min()) / (image_raw_np.max() - image_raw_np.min())
    )
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask_raw_np, cmap="gray")
    axs[0, 1].set_title("Original Mask")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(
        (image_aug_np - image_aug_np.min()) / (image_aug_np.max() - image_aug_np.min())
    )
    axs[1, 0].set_title("Augmented Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mask_aug_np, cmap="gray")
    axs[1, 1].set_title("Augmented Mask")
    axs[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / f"compare_{idx + 1}.png", dpi=150)
    plt.close()

print(f"Saved {n_samples} 2x2 comparison samples to {output_dir}/")
