#train.py

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import (
    build_consensus_matrix,
    consensus_consistency_loss, get_base_clusterings
)


def train_autoencoder(dataloader, model, use_gpu, **kwargs):
    """
    Train the autoencoder model.

    Parameters:
    dataloader (DataLoader): Dataloader for the training data.
    model (nn.Module): Autoencoder model to train.
    use_gpu (bool): Flag to indicate whether to use GPU if available.
    num_epochs (int): Number of epochs to train the model.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    running_loss = 0.0
    for batch in dataloader:
        if batch is None:
            continue

        # Flexible unpacking (2 or 3 elements)
        if len(batch) == 3:
            images, _, _ = batch
        elif len(batch) == 2:
            images, _ = batch
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")

        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def evaluate_autoencoder(dataloader, model, use_gpu):
    """
    Evaluate the autoencoder model on the test set.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in dataloader:
            if (images, _) is None:
                continue
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss



def train_constrained_autoencoder(
    dataloader,
    model,
    use_gpu,
    num_epochs=4,
    tag_tuner=0.5,
    consensus_matrix=None,
    lambda_consensus=0.0,
    **kwargs
):
    """
    Train a constrained autoencoder with optional DECCS global consensus loss.

    Args:
        dataloader (DataLoader): Training data loader.
        model (nn.Module): ConstrainedAutoencoder instance.
        use_gpu (bool): Whether to use GPU.
        num_epochs (int): Number of epochs.
        tag_tuner (float): Weight for tag supervision loss.
        consensus_matrix (np.ndarray or None): Global consensus matrix (optional).
        lambda_consensus (float): Weight for consensus consistency loss.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    recon_loss_fn = nn.MSELoss()
    tag_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    all_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            # Unpack batch
            images, symbolic_tags, _ = batch
            images = images.to(device)
            symbolic_tags = symbolic_tags.to(device)

            # Forward pass
            recon, tag_logits = model(images)
            recon_loss = recon_loss_fn(recon, images)
            tag_loss = tag_loss_fn(tag_logits, symbolic_tags)

            # Consensus loss (if global consensus provided)
            if consensus_matrix is not None and lambda_consensus > 0:
                with torch.no_grad():
                    # Get embeddings for this batch (for alignment)
                    z = model.get_embeddings(images).detach()
                consensus_loss = consensus_consistency_loss(z, consensus_matrix)
                total_loss = recon_loss + tag_tuner * tag_loss + lambda_consensus * consensus_loss
            else:
                total_loss = recon_loss + tag_tuner * tag_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(dataloader)
        all_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed. "
                     f"Recon={recon_loss.item():.4f}, Tag={tag_loss.item():.4f}, "
                     f"Total={epoch_loss:.4f}")

        if torch.isnan(total_loss):
            logging.error("NaN detected in total_loss — check tag inputs or consensus matrix normalization.")
            continue

    return all_losses
