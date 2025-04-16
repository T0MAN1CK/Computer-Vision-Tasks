import torch
import wandb
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.dataset import SKU110KDataset
from DETR.lightning_module import DETRLightningModule
from shared.test_utils import evaluate_map_metrics, log_worst_predictions
from DETR.model.detr import DETR

base_dir = Path(__file__).resolve().parent.parent
ckpt_path = base_dir / "DETR" / "checkpoints" / "best-v1.ckpt"


def collate_fn(batch):
    images, targets = zip(*batch)  # images: tuple of tensors
    return list(images), list(targets)  # both lists


def main():
    torch.set_float32_matmul_precision("medium")
    pl_seed = 42
    torch.manual_seed(pl_seed)

    # --- WandB Init ---
    wandb.init(
        project="mod3_testing",
        name="DETR_Eval",
        job_type="eval",
        group="DETR",
    )

    # --- Load model ---
    detr_model = DETR(
        num_classes=2,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    )

    model = DETRLightningModule.load_from_checkpoint(ckpt_path, model=detr_model)
    model.eval().cuda()

    # --- Dataset & Loader ---
    test_dataset = SKU110KDataset(
        csv_path=base_dir / "dataset/sample_SKU110K/annotations/annotations_test.csv",
        image_dir=base_dir / "dataset/sample_SKU110K/images",
        use_aug=False,
        visualize=False,
        resize_to=(800, 800),
        model_type="detr",
    )

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # --- Evaluate & Log ---
    results = evaluate_map_metrics(model, loader)
    log_worst_predictions(results, count=10)

    wandb.finish()


if __name__ == "__main__":
    main()
