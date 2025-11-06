#train.py

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import consensus_consistency_loss

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

def make_grad_scaler(device):
    """
    Version-safe GradScaler initialization.
    Uses new torch.amp API if supported, else falls back to legacy CUDA AMP.
    """
    try:
        return GradScaler(device_type=device, enabled=(device == "cuda"))
    except TypeError:        
        return GradScaler(enabled=(device == "cuda"))

def _prepare_model_for_cuda(model):
    """
    Safely convert convolutional layers to channels_last memory format for GPU speedup.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.to(memory_format=torch.channels_last)
    return model

def train_autoencoder(dataloader, model, use_gpu, num_epochs=1, **kwargs):
    """
    Train the autoencoder model.

    Parameters:
    dataloader (DataLoader): Dataloader for the training data.
    model (nn.Module): Autoencoder model to train.
    use_gpu (bool): Flag to indicate whether to use GPU if available.
    num_epochs (int): Number of epochs to train the model.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        model = _prepare_model_for_cuda(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device.type)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            if batch is None:
                continue

            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"[AE] Epoch {epoch + 1}/{num_epochs} | Loss={avg_loss:.6f}")

    return avg_loss

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
    model.to(device)
    if device.type == 'cuda':
        model = _prepare_model_for_cuda(model)

    recon_loss_fn = nn.MSELoss()
    tag_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device.type)
    all_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            if batch is None:
                continue

            images, symbolic_tags, idx = batch
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            symbolic_tags = symbolic_tags.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                recon, tag_logits = model(images)
                recon_loss = recon_loss_fn(recon, images)
                tag_loss = tag_loss_fn(tag_logits, symbolic_tags)

                if consensus_matrix is not None and lambda_consensus > 0:
                    with torch.no_grad():
                        z = model.get_embeddings(images)
                    if isinstance(idx, torch.Tensor):
                        idx = idx.cpu().numpy()
                    consensus_sub = consensus_matrix[np.ix_(idx, idx)]
                    consensus_loss = consensus_consistency_loss(z, consensus_sub)
                    total_loss = recon_loss + tag_tuner * tag_loss + lambda_consensus * consensus_loss
                else:
                    total_loss = recon_loss + tag_tuner * tag_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.detach().item()

        if 'loss_log' not in locals():
            loss_log = {"total": [], "recon": [], "tag": []}
        loss_log["total"].append(total_loss.item())
        loss_log["recon"].append(recon_loss.item())
        loss_log["tag"].append(tag_loss.item())

        epoch_loss = running_loss / len(dataloader)
        all_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed. "
                    f"Recon={recon_loss.item():.4f}, Tag={tag_loss.item():.4f}, "
                    f"Total={epoch_loss:.4f}")

        if torch.isnan(total_loss):
            logging.error("NaN detected in total_loss — check tag inputs or consensus matrix normalization.")
            continue
        np.savez("results_deccs_loss_components.npz",
                 total=np.array(loss_log["total"]),
                 recon=np.array(loss_log["recon"]),
                 tag=np.array(loss_log["tag"]))

    return all_losses
