#train.py

import logging
import torch
import torch.nn as nn
import torch.optim as optim


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
    total_loss = 0.0
    for images, _ in dataloader:
        if (images, _) is None:
            continue
        images = images.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
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
    tag_criterion = nn.BCEWithLogitsLoss()  # For tag prediction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    running_loss = 0.0
    tag_tuner = 0.5
    for images, symbolic_tags in dataloader:
        if (images, symbolic_tags) is None:
            continue
        images = images.to(device)
        symbolic_tags = symbolic_tags.to(device)

        # Forward pass through the autoencoder
        outputs, predicted_tags = model(images)  # Get both image reconstruction and tag predictions
        
        # Reconstruction loss (for images)
        recon_loss = criterion(outputs, images)
        logging.info("reconstruction loss {recon_loss}")

        # Tag prediction loss
        tag_loss = tag_criterion(predicted_tags, symbolic_tags)
        logging.info("tag loss {tag_loss}")

        # Total loss combines reconstruction loss and tag loss
        total_loss = recon_loss + tag_tuner * tag_loss
        logging.info("total loss {total_loss} (reminder : tag_tuner might need adjustment (0.5-2))")

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        # ToDo add early stopping (torch.optim.lr_scheduler.StepLR)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss
