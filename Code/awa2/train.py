#train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_autoencoder(dataloader, model, use_gpu):
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
    for images, _ in dataloader:
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def train_constrained_autoencoder(dataloader, model, use_gpu):
    """
    Train the autoencoder model.

    Parameters:
    dataloader (DataLoader): Dataloader for the training data.
    model (nn.Module): Autoencoder model to train.
    use_gpu (bool): Flag to indicate whether to use GPU if available.
    num_epochs (int): Number of epochs to train the model.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()  # For reconstruction loss
    tag_criterion = nn.MSELoss()  # For tag prediction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    running_loss = 0.0
    for images, symbolic_tags in dataloader:
        images = images.to(device)
        symbolic_tags = symbolic_tags.to(device)

        # Forward pass through the autoencoder
        outputs, predicted_tags = model(images)  # Get both image reconstruction and tag predictions
        
        # Reconstruction loss (for images)
        recon_loss = criterion(outputs, images)
        
        # Tag prediction loss
        tag_loss = tag_criterion(predicted_tags, symbolic_tags)

        # Total loss combines reconstruction loss and tag loss
        total_loss = recon_loss + tag_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss
