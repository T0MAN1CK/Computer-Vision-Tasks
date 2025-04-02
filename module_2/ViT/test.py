import torch
import glob
import wandb
from torchvision.transforms.functional import to_pil_image

from models.model import ViTModule
from data.datamodule import ImageClassificationDataModule
from data.dataset import CLASS_NAMES


def evaluate_metrics(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def log_misclassified_images(model, dataloader, limit=25):
    model.eval()
    misclassified = []

    with torch.no_grad():
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


def main():
    wandb.init(
        project="vit-classification",
        name="test-run",
        dir="module_2/ViT/wandblogs",
        job_type="eval",
    )

    ckpt_list = glob.glob("module_2/ViT/checkpoints/*.ckpt")
    assert len(ckpt_list) == 1, "Expected exactly one checkpoint"
    model = ViTModule.load_from_checkpoint(ckpt_list[0])
    model.cuda()
    model.eval()

    datamodule = ImageClassificationDataModule(
        data_dir="module_2/Classification_data", batch_size=64
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    test_loss, test_acc = evaluate_metrics(model, test_loader)
    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc})
    wandb.summary.update(
        {"final_test_loss": test_loss, "final_test_accuracy": test_acc}
    )

    log_misclassified_images(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
