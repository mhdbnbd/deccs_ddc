import torch
import torch.nn as nn
import torch.optim as optim


def train_autoencoder(dataloader, model, use_gpu, num_epochs=10):
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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _, _, _ in dataloader:
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
