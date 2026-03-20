"""
Model architectures for DECCS-DDC integration.

Three architecture tiers:
  - small:  Original 4-layer CNN autoencoder (128x128) — baseline
  - resnet: Frozen ResNet-101 + trainable MLP — DDC-style (main contribution)
  - large:  Deeper CNN autoencoder (224x224) — ablation

The ResNet model (DDCNet) implements DDC's architecture:
  - Frozen pretrained backbone (ResNet-101, matching DDC paper)
  - Trainable 3-layer MLP producing cluster assignment logits
  - Tag prediction head for pairwise constraint generation
  - No decoder / no reconstruction loss

References:
  DDC:   Zhang & Davidson, "Deep Descriptive Clustering", IJCAI 2021
  DECCS: Miklautz et al., "Deep Clustering with Consensus Representations", ICDM 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Original small 4-layer CNN autoencoder (baseline, 128x128)
# =============================================================================

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embeddings(self, x):
        z = self.encoder(x)
        return F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)


class ConstrainedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc_tags = nn.Linear(128, 85)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(encoded, 1).view(encoded.size(0), -1)
        return self.decoder(encoded), self.fc_tags(pooled)

    def get_embeddings(self, x):
        z = self.encoder(x)
        return F.adaptive_avg_pool2d(z, 1).view(z.size(0), -1)


# =============================================================================
# DDC-style ResNet-101 + MLP (main contribution)
#
# Architecture follows DDC (Zhang & Davidson, 2021):
#   Frozen ResNet-101 → 3-layer MLP → K-dim softmax cluster assignments
#   + tag prediction head from hidden layer (for pairwise constraints)
#   + no decoder, no reconstruction
#
# Training (in train.py):
#   L = L_MI + lambda_pairwise * L_pairwise [+ lambda_consensus * L_consensus]
#   where L_MI = mutual information clustering loss
#         L_pairwise = KL divergence between cluster assignments of tag-similar pairs
#         L_consensus = DECCS ensemble agreement (our extension)
# =============================================================================

class DDCNet(nn.Module):
    """
    DDC-style network: frozen ResNet-101 → MLP → cluster assignments.

    forward() returns dict:
      'cluster_probs':  (N, K) softmax cluster probabilities
      'tag_logits':     (N, T) tag prediction logits
      'embeddings':     (N, hidden_dim) hidden representations for consensus
    """

    def __init__(self, n_clusters=50, n_attributes=85, hidden_dim=1200):
        super().__init__()
        self.n_clusters = n_clusters

        # Frozen ResNet-101
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DDC's 3-layer MLP (trainable)
        self.mlp = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        self.tag_head = nn.Linear(hidden_dim, n_attributes)

    def forward(self, x):
        """Full forward: images → ResNet → MLP → outputs."""
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        return self.forward_from_features(features)

    def forward_from_features(self, features):
        """Forward from pre-extracted ResNet features (skips backbone)."""
        hidden = self.mlp(features)
        cluster_logits = self.cluster_head(hidden)
        return {
            'cluster_probs': F.softmax(cluster_logits, dim=1),
            'tag_logits': self.tag_head(hidden),
            'embeddings': hidden,
        }

    def extract_backbone_features(self, dataloader, device):
        """Pre-extract frozen ResNet features for the entire dataset."""
        from tqdm import tqdm
        self.eval()
        all_features, all_tags, all_indices = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting ResNet features"):
                if batch is None:
                    continue
                images, tags, idx = batch
                features = self.backbone(images.to(device)).flatten(1)
                all_features.append(features.cpu())
                all_tags.append(tags)
                all_indices.append(idx)
        self.train()
        return (torch.cat(all_features),
                torch.cat(all_tags),
                torch.cat(all_indices))

    def get_embeddings(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(1)
        return self.mlp(features)

    def get_cluster_assignments(self, x):
        return self.forward(x)['cluster_probs'].argmax(dim=1)


# Aliases for backward compatibility
ResNetAutoencoder = DDCNet
ResNetConstrainedAutoencoder = DDCNet


# =============================================================================
# Large CNN autoencoder (ablation, 224x224)
# =============================================================================

class LargeAutoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc_embed = nn.Linear(512, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)
        return self.decoder(self.fc_embed(z))

    def get_embeddings(self, x):
        z = F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)
        return self.fc_embed(z)


class LargeConstrainedAutoencoder(nn.Module):
    def __init__(self, embedding_dim=128, n_attributes=85):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc_embed = nn.Linear(512, embedding_dim)
        self.fc_tags = nn.Linear(embedding_dim, n_attributes)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)
        z = self.fc_embed(z)
        return self.decoder(z), self.fc_tags(z)

    def get_embeddings(self, x):
        z = F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)
        return self.fc_embed(z)