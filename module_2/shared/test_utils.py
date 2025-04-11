import torch
import wandb
from typing import Tuple
from torchvision.transforms.functional import to_pil_image
from shared.constants import CLASS_NAMES


@torch.inference_mode()
def evaluate_metrics(model: torch.nn.Module, dataloader) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for images, targets in dataloader:
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        loss = criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
        total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.inference_mode()
def log_misclassified_images(model: torch.nn.Module, dataloader, limit: int = 25):
    model.eval()
    misclassified = []

    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for img, true, pred in zip(images, labels, preds):
            if true != pred and len(misclassified) < limit:
                misclassified.append(
                    (img.cpu(), CLASS_NAMES[true.item()], CLASS_NAMES[pred.item()])
                )

        if len(misclassified) >= limit:
            break

    if misclassified:
        wandb.log(
            {
                "misclassified_examples": [
                    wandb.Image(
                        to_pil_image(img), caption=f"True: {true}, Pred: {pred}"
                    )
                    for img, true, pred in misclassified
                ]
            }
        )
