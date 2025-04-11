import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import wandb
from pathlib import Path
from pytorch_lightning import seed_everything

from FasterRCNN.lightning_module import FasterRCNNLightningModule
from FasterRCNN.data.dataset import SKU110KDataset
from FasterRCNN.utils.test_utils import evaluate_map_metrics, log_worst_predictions


def main():
    torch.set_float32_matmul_precision("medium")
    seed_everything(42)

    base_dir = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / "FasterRCNN" / "checkpoints" / "best.ckpt"
    ann_path = (
        base_dir / "dataset" / "sample_SKU110K" / "annotations" / "annotations_test.csv"
    )
    img_dir = base_dir / "dataset" / "sample_SKU110K" / "images"
    logs_dir = base_dir / "FasterRCNN" / "logs"

    wandb.init(
        project="module_3",
        name="FasterRCNN_Eval",
        dir=str(logs_dir),
        job_type="eval",
        group="FasterRCNN",
    )

    model = FasterRCNNLightningModule.load_from_checkpoint(str(ckpt_path)).cuda().eval()

    dataset = SKU110KDataset(
        csv_path=ann_path,
        image_dir=img_dir,
        use_aug=False,
        visualize=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    results = evaluate_map_metrics(model, loader)
    log_worst_predictions(results, count=10)

    wandb.finish()


if __name__ == "__main__":
    main()
