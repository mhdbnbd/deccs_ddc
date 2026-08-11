"""DDC encoder network.

Paper section 4.1 (Implementations) specifies, for all clustering tasks, frozen
pre-trained ResNet-101 features and an encoder of three fully connected layers
sized [1200, 1200, K], with ReLU activations.

So: 2048 -> 1200 -> ReLU -> 1200 -> ReLU -> K -> softmax. The backbone is frozen
and its features are pre-extracted, so it is not part of this module.

The paper states nothing about weight initialization, normalization layers,
dropout or softmax temperature. JUDGMENT CALLS: PyTorch default nn.Linear init,
no BatchNorm/LayerNorm/dropout, temperature fixed at 1.0 (exposed as an argument
for ablation only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDCEncoder(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=1200, n_clusters=50, temperature=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_clusters),
        )

    def forward(self, x):
        """x: (B, in_dim) pre-extracted frozen features -> (B, K) probabilities."""
        return F.softmax(self.net(x) / self.temperature, dim=1)

    @torch.no_grad()
    def predict_probs(self, x, device, batch_size=4096):
        """Full-dataset inference in chunks. x is a torch tensor on CPU."""
        self.eval()
        out = []
        for start in range(0, x.shape[0], batch_size):
            out.append(self.forward(x[start:start + batch_size].to(device)).cpu())
        self.train()
        return torch.cat(out)