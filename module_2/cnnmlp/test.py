import glob
import torch
import wandb
from pathlib import Path
import torch.nn.functional as F
from models.cnn_mlp import CNNMLPClassifier
from data.dataset import ClassificationDataModule, CLASS_NAMES


def log_test_metrics(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    wandb.log({"test/loss": avg_loss, "test/accuracy": avg_acc})

    return avg_loss, avg_acc


def log_misclassified_images(model, dataloader, num_images=25):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(imgs)):
                if preds[i] != labels[i] and len(misclassified) < num_images:
                    misclassified.append(
                        (imgs[i], CLASS_NAMES[labels[i]], CLASS_NAMES[preds[i]])
                    )

    if misclassified:
        images = [img.cpu() for img, _, _ in misclassified]
        captions = [f"True: {true}, Pred: {pred}" for _, true, pred in misclassified]

        wandb.log(
            {
                "misclassified_examples": [
                    wandb.Image(img, caption=caption)
                    for img, caption in zip(images, captions)
                ]
            }
        )


def main():
    current_dir = Path(__file__).parent.absolute()
    checkpoint_dir = current_dir / "trained_models"
    data_dir = current_dir.parent / "Classification_data"
    logs_dir = current_dir / "logs"

    wandb.init(
        project="cnnmlp-classification",
        name="test-run",
        dir=str(logs_dir),
        job_type="eval",
    )

    ckpt_list = glob.glob(str(checkpoint_dir / "*.ckpt"))
    assert len(ckpt_list) == 1, "Expected exactly one checkpoint"
    ckpt_path = ckpt_list[0]

    model = CNNMLPClassifier.load_from_checkpoint(ckpt_path, num_classes=6).cuda()

    data_module = ClassificationDataModule(data_dir=str(data_dir), batch_size=64)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    test_loss, test_acc = log_test_metrics(model, test_loader)
    log_misclassified_images(model, test_loader)

    wandb.summary.update(
        {"final_test_loss": test_loss, "final_test_accuracy": test_acc}
    )

    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
