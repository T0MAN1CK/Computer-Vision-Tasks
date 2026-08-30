# ruff: noqa

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("UNET/checkpoints", exist_ok=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
import copy
from tqdm import tqdm
import scipy.ndimage as ndi
import numpy as np

from UNET.model import build_model
from UNET.losses import criterion, dice_coef, iou_coef
from UNET.dataloader import prepare_loaders


def count_blobs(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    _, num_blobs = ndi.label(binary_mask)
    return num_blobs


def compute_smape(gt_counts, pred_counts):
    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    denominator = (np.abs(gt_counts) + np.abs(pred_counts)) / 2.0
    return np.mean(np.abs(gt_counts - pred_counts) / (denominator + 1e-6)) * 100


class CFG:
    fold = 0
    epochs = 50
    lr = 2e-3
    min_lr = 1e-6
    batch_size = 16
    scheduler = "CosineAnnealingLR"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    project = "module_4_segmentation"
    run_name = "UNET-train-count"
    image_size = (256, 256)
    accum = 1
    wd = 1e-6
    csv_path = "segmentation_dataset/train_folds.csv"


cfg = CFG()


def fetch_scheduler(optimizer):
    if cfg.scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, verbose=True
        )
    return None


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    optimizer.zero_grad()

    for step, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks) / cfg.accum

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum == 0 or (step + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler and cfg.scheduler != "ReduceLROnPlateau":
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        pbar.set_postfix(loss=running_loss / dataset_size)

    return running_loss / dataset_size


@torch.no_grad()
def valid_one_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_dice, all_iou = [], []
    all_gt_counts, all_pred_counts = [], []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")
    for step, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        preds = torch.sigmoid(outputs)
        preds_bin = (preds > 0.5).float()

        all_dice.append(dice_coef(masks, preds_bin).item())
        all_iou.append(iou_coef(masks, preds_bin).item())

        for i in range(batch_size):
            gt_mask_np = masks[i].cpu().squeeze().numpy() > 0.5
            pred_mask_np = preds_bin[i].cpu().squeeze().numpy() > 0.5

            gt_num = int(ndi.label(gt_mask_np)[1])
            pred_num = int(ndi.label(pred_mask_np)[1])

            all_gt_counts.append(gt_num)
            all_pred_counts.append(pred_num)

            img = images[i].cpu().permute(1, 2, 0).numpy()
            pred = preds_bin[i].cpu().squeeze().numpy()
            gt = masks[i].cpu().squeeze().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            wandb.log(
                {
                    "val/image": wandb.Image(img, caption="Input"),
                    "val/gt_mask": wandb.Image(gt, caption=f"GT Count: {gt_num}"),
                    "val/pred_mask": wandb.Image(
                        pred, caption=f"Pred Count: {pred_num}"
                    ),
                }
            )

    val_loss = running_loss / dataset_size
    val_dice = sum(all_dice) / len(all_dice)
    val_iou = sum(all_iou) / len(all_iou)
    val_smape = compute_smape(all_gt_counts, all_pred_counts)

    wandb.log(
        {
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_smape": val_smape,
        }
    )

    return val_loss, val_dice, val_iou, val_smape


def run_training():
    wandb.init(project=cfg.project, name=cfg.run_name, config=vars(cfg))
    train_loader, val_loader = prepare_loaders(
        cfg.csv_path, cfg.fold, cfg.batch_size, num_workers=4, debug=False
    )

    model = build_model().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = fetch_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler()

    best_smape = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 7
    early_stop_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg.device
        )

        if epoch % 2 == 0:
            val_loss, val_dice, val_iou, val_smape = valid_one_epoch(
                model, val_loader, cfg.device
            )

            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_smape": val_smape,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            print(
                f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Val SMAPE: {val_smape:.2f}%"
            )

            if val_smape < best_smape:
                best_smape = val_smape
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    model.state_dict(), f"UNET/checkpoints/best_fold{cfg.fold}.pt"
                )
                print("Saved best model")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            wandb.log({"train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]})

    torch.save(best_model_wts, f"UNET/checkpoints/final_fold{cfg.fold}.pt")
    wandb.finish()


if __name__ == "__main__":
    run_training()
