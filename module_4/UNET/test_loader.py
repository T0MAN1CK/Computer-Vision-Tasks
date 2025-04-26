# ruff: noqa

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add root directory

import matplotlib.pyplot as plt
import warnings

from UNET.dataloader import prepare_loaders

warnings.filterwarnings(
    "ignore", message="Default grid_sample and affine_grid behavior"
)

import gc
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_loader, val_loader = prepare_loaders(
    csv_path="segmentation_dataset/train_folds.csv",
    fold=0,
    batch_size=8,
    num_workers=0,
    debug=True,
)

images, masks = next(iter(train_loader))


def plot_batch(imgs, msks, size=5):
    plt.figure(figsize=(5 * size, 5))
    for idx in range(size):
        plt.subplot(2, size, idx + 1)
        img = imgs[idx].permute(1, 2, 0).numpy()
        plt.imshow((img - img.min()) / (img.max() - img.min()))
        plt.axis("off")

        plt.subplot(2, size, idx + 1 + size)
        plt.imshow(msks[idx].squeeze().numpy(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


plot_batch(images, masks, size=5)
gc.collect()
