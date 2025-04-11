import torch
import torch.nn as nn
import pytorch_lightning as pl


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=144):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, emb_size, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x + self.pos_embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, expansion=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=144,
        depth=6,
        n_classes=6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(emb_size) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class ViTModule(pl.LightningModule):
    def __init__(self, num_classes=6, lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViT(n_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        return [optimizer], [scheduler]
