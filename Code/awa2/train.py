import torch
from torch import nn, optim


def train_autoencoder(dataloader, autoencoder, use_gpu):
    """
    Function to train the autoencoder.
    Args:
    - dataloader (DataLoader): DataLoader for the dataset.
    - autoencoder (nn.Module): Autoencoder model.
    - use_gpu (bool): Flag to determine whether to use GPU for training.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, labels, attributes) in enumerate(dataloader):
            if images is None or labels is None or attributes is None:
                print(f"Skipping batch {i} due to None values")
                continue

            if use_gpu:
                images = images.cuda()

            # Forward pass
            outputs = autoencoder(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
