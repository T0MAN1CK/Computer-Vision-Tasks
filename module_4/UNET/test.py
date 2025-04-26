# ruff: noqa

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import json
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from scipy.ndimage import label
import cv2
import matplotlib.pyplot as plt
import wandb

from UNET.model import load_model

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "UNET/checkpoints/final_fold0.pt"
TEST_DIR = Path("segmentation_dataset/test_data")
LABELS_PATH = TEST_DIR / "test_labels_downscaled.json"
PATCH_SIZE, STRIDE = 256, 128
TARGET_SIZE = (768, 1024)

wandb.init(project="module_4_segmentation", name="UNET-test")

model = load_model(CHECKPOINT_PATH, device=DEVICE)
model.eval()

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


def extract_patches(image, patch_size=256, stride=128):
    C, H, W = image.shape
    unfold = F.unfold(image.unsqueeze(0), kernel_size=patch_size, stride=stride)
    patches = unfold.squeeze(0).transpose(0, 1).reshape(-1, C, patch_size, patch_size)
    return patches, H, W


def combine_patches(patches, H, W, patch_size=256, stride=128):
    C = patches.shape[1]
    patches_flat = patches.reshape(patches.shape[0], -1).transpose(0, 1)
    fold = F.fold(
        patches_flat.unsqueeze(0),
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride,
    )
    norm_map = F.fold(
        torch.ones_like(patches_flat).unsqueeze(0),
        output_size=(H, W),
        kernel_size=patch_size,
        stride=stride,
    )
    return (fold / norm_map).squeeze(0)


def compute_smape(gt, pred):
    gt, pred = np.array(gt), np.array(pred)
    denominator = (np.abs(gt) + np.abs(pred)) / 2
    smape = np.mean(np.abs(gt - pred) / (denominator + 1e-6)) * 100
    return smape


def watershed_count(binary_mask):
    binary_mask = np.squeeze(binary_mask).astype(np.uint8)
    three_channel_img = np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.3 * distance.max(), 1, 0)
    sure_fg = sure_fg.astype(np.uint8)
    markers, _ = label(sure_fg)
    markers = markers.astype(np.int32)
    markers = cv2.watershed(three_channel_img, markers)
    unique_labels = np.unique(markers)
    object_count = len(unique_labels[(unique_labels > 1)])
    return object_count


gt_counts_all, pred_counts_all = [], []

for key, entry in labels.items():
    filename = entry["filename"]
    image_path = TEST_DIR / filename
    image = read_image(str(image_path)).float() / 255.0
    image = TF.resize(image, TARGET_SIZE)

    patches, H, W = extract_patches(image, PATCH_SIZE, STRIDE)
    preds = []
    with torch.no_grad():
        for patch in patches:
            patch = patch.unsqueeze(0).to(DEVICE)
            pred = torch.sigmoid(model(patch)).cpu().squeeze(0)
            preds.append(pred)
    preds = torch.stack(preds)
    pred_mask = combine_patches(preds, H, W, PATCH_SIZE, STRIDE)

    pred_binary = (pred_mask.numpy() > 0.5).astype(np.uint8)
    pred_count = watershed_count(pred_binary)
    gt_count = len(entry["regions"])

    gt_counts_all.append(gt_count)
    pred_counts_all.append(pred_count)

    wandb.log({f"{filename}/gt_count": gt_count, f"{filename}/pred_count": pred_count})

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image.permute(1, 2, 0).numpy())
    axs[0].set_title(f"Original: {filename}")

    axs[1].imshow(pred_mask.squeeze(0).numpy(), cmap="gray")
    axs[1].set_title("Predicted Mask")

    axs[2].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    axs[2].imshow(pred_mask.squeeze(0).numpy(), cmap="Reds", alpha=0.5)
    axs[2].set_title("Overlay")

    for ax in axs:
        ax.axis("off")

    wandb.log({f"{filename}": wandb.Image(plt)})
    plt.close()

smape = compute_smape(gt_counts_all, pred_counts_all)
wandb.log({"test_smape": smape})
print(f"Inference complete | SMAPE: {smape:.2f}%")
