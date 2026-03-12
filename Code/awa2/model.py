import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Option 0: Original small 4-layer CNN (1.2M params, 128x128 input)
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # (N, 3, 128, 128) → (N, 16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (N, 16, 64, 64) → (N, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (N, 32, 32, 32) → (N, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # (N, 64, 16, 16) → (N, 128, 8, 8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_embeddings(self, x):
        """Return pooled latent embeddings for clustering."""
        z = self.encoder(x)
        z = F.adaptive_avg_pool2d(z, 1)
        return z.view(z.size(0), -1)


class ConstrainedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_tags = nn.Linear(128, 85)  # Predict symbolic tags (85 dims)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(encoded, 1)
        pooled = pooled.view(pooled.size(0), -1)
        predicted_tags = self.fc_tags(pooled)
        decoded = self.decoder(encoded)
        return decoded, predicted_tags

    def get_embeddings(self, x):
        """Return latent embeddings used for clustering or consensus loss."""
        z = self.encoder(x)
        z = F.adaptive_avg_pool2d(z, 1)
        return z.view(z.size(0), -1)


# =============================================================================
# Pretrained ResNet-101 backbone (~44M params, 224x224 input)
# Frozen backbone extracts rich features; only the projection head,
# tag predictor, and decoder are trained.
# ResNet-101 is used to match DDC's experimental setup (Zhang & Davidson, 2021).
# =============================================================================

class ResNetAutoencoder(nn.Module):
    """Autoencoder using frozen pretrained ResNet-101 as encoder."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Use everything up to avgpool as the frozen feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (N, 2048, 1, 1)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable projection from ResNet features to embedding space
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        # Lightweight decoder from embedding (for reconstruction loss)
        # Upsamples from embedding_dim to 224x224x3
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # → 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),     # → 224x224
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(1)  # (N, 2048)
        z = self.projector(features)                  # (N, embedding_dim)
        decoded = self.decoder(z)
        return decoded

    def get_embeddings(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        return self.projector(features)


class ResNetConstrainedAutoencoder(nn.Module):
    """Constrained autoencoder using frozen pretrained ResNet-101 as encoder."""

    def __init__(self, embedding_dim=128, n_attributes=85):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (N, 2048, 1, 1)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

        # Tag prediction head
        self.fc_tags = nn.Linear(embedding_dim, n_attributes)

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # → 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),     # → 224x224
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(1)   # (N, 2048)
        z = self.projector(features)                  # (N, embedding_dim)
        predicted_tags = self.fc_tags(z)
        decoded = self.decoder(z)
        return decoded, predicted_tags

    def get_embeddings(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        return self.projector(features)


# =============================================================================
# Larger custom CNN (8-layer, ~5M params, 224x224 input)
# Trains from scratch — more capacity than original but no pretrained knowledge.
# Uses BatchNorm for training stability at deeper depth.
# =============================================================================

class LargeAutoencoder(nn.Module):
    """Deeper CNN autoencoder for 224x224 input."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # → 112x112
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 56x56
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # → 28x28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # → 14x14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # → 7x7
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc_embed = nn.Linear(512, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # → 14x14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # → 28x28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → 56x56
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → 112x112
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # → 224x224
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        z = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        z = self.fc_embed(z)
        decoded = self.decoder(z)
        return decoded

    def get_embeddings(self, x):
        z = self.encoder(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc_embed(z)


class LargeConstrainedAutoencoder(nn.Module):
    """Deeper CNN constrained autoencoder for 224x224 input."""

    def __init__(self, embedding_dim=128, n_attributes=85):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # → 112x112
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 56x56
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # → 28x28
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # → 14x14
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # → 7x7
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc_embed = nn.Linear(512, embedding_dim)
        self.fc_tags = nn.Linear(embedding_dim, n_attributes)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        z = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        z = self.fc_embed(z)
        predicted_tags = self.fc_tags(z)
        decoded = self.decoder(z)
        return decoded, predicted_tags

    def get_embeddings(self, x):
        z = self.encoder(x)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc_embed(z)