import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):
        super().__init__()
        # Pretrained ResNet-50 backbone
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])

        # 1x1 Conv to project ResNet output to transformer dimension
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # Learnable object queries
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.linear_bbox = nn.Linear(hidden_dim, 4)

    def forward(self, inputs: torch.Tensor):
        x = self.backbone(inputs)
        h = self.conv(x)

        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        h_flat = h.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        queries = self.query_pos.unsqueeze(1).repeat(1, inputs.size(0), 1)

        hs = self.transformer(h_flat + pos, queries)  # [num_queries, B, C]

        outputs_class = self.linear_class(hs)
        outputs_bbox = self.linear_bbox(hs).sigmoid()

        return outputs_class.transpose(0, 1), outputs_bbox.transpose(0, 1)
