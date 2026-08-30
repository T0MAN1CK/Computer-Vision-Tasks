import torch
import segmentation_models_pytorch as smp


def build_model(encoder_name="timm-resnest50d", in_channels=3, out_classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_classes,
        activation=None,  # Apply sigmoid manually in validation
    )
    return model


def load_model(path, device="cuda"):
    model = build_model()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)
