# ruff: noqa: E402

import torch
import os
import sys
import wandb
import glob
import torch.nn.functional as F

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from finetuned_cnn.models.model import FinetunedResNet
from finetuned_cnn.datamodule import ClassificationDataModule
from finetuned_cnn.data.dataset import CLASS_NAMES


def evaluate_metrics(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to("cuda")
            targets = targets.to("cuda")
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def log_misclassified_images(model, dataloader):
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images = images.to("cuda")
            targets = targets.to("cuda")
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            mismatches = preds != targets

            misclassified_images.extend(images[mismatches].cpu())
            misclassified_preds.extend(preds[mismatches].cpu())
            misclassified_targets.extend(targets[mismatches].cpu())

    logged_images = []
    for img, pred, target in zip(
        misclassified_images, misclassified_preds, misclassified_targets
    ):
        img = img.permute(1, 2, 0).numpy()
        caption = f"True: {CLASS_NAMES[target]} | Pred: {CLASS_NAMES[pred]}"
        logged_images.append(wandb.Image(img, caption=caption))

    if logged_images:
        wandb.log({"misclassified_examples": logged_images})


def main():
    wandb.init(
        project="finetuned-cnn-classification",
        name="test-run",
        dir="module_2/finetuned_cnn/wandblogs",
        job_type="eval",
    )

    ckpt_list = glob.glob("module_2/finetuned_cnn/checkpoints/*.ckpt")
    assert len(ckpt_list) == 1, "Expected exactly one checkpoint"
    model = FinetunedResNet.load_from_checkpoint(ckpt_list[0])
    model.cuda()
    model.eval()

    datamodule = ClassificationDataModule(
        data_dir="module_2/Classification_data", batch_size=64
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # Evaluate and log
    test_loss, test_acc = evaluate_metrics(model, test_loader)
    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc})
    wandb.summary.update(
        {"final_test_loss": test_loss, "final_test_accuracy": test_acc}
    )

    # Log misclassified examples
    log_misclassified_images(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
