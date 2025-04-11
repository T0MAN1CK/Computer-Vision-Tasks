# ruff: noqa: E402

import torch
import wandb
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from shared.test_utils import evaluate_metrics, log_misclassified_images
from shared.datamodule import UniversalDataModule
from shared.litmodule import LitClassifier
from models.cnn_mlp import CNNMLPClassifier


def main():
    torch.set_float32_matmul_precision("medium")

    base_dir = Path(__file__).parent.parent
    ckpt_dir = base_dir / "cnnmlp" / "trained_models"
    data_dir = base_dir / "Classification_data"
    logs_dir = base_dir / "cnnmlp" / "logs"

    ckpt_list = glob.glob(str(ckpt_dir / "*.ckpt"))
    assert len(ckpt_list) == 1, "Expected exactly one checkpoint"

    wandb.init(
        project="image-classification",
        name="cnnmlp-eval",
        dir=str(logs_dir),
        job_type="eval",
        group="CNNMLP",
    )

    model = CNNMLPClassifier(num_classes=6)
    lit_model = (
        LitClassifier.load_from_checkpoint(
            ckpt_list[0],
            model=model,
            num_classes=6,
            lr=1e-3,
            weight_decay=1e-4,
        )
        .cuda()
        .eval()
    )

    datamodule = UniversalDataModule(
        data_dir=str(data_dir),
        batch_size=64,
        val_split=0.1,
        use_kornia_aug=False,
    )
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()
    test_loss, test_acc = evaluate_metrics(lit_model, test_loader)

    wandb.log(
        {
            "test/loss": test_loss,
            "test/accuracy": test_acc,
        }
    )
    wandb.summary.update(
        {
            "final_test_loss": test_loss,
            "final_test_accuracy": test_acc,
        }
    )

    log_misclassified_images(lit_model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    main()
