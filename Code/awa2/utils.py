import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    """
    Custom collate function to filter out None values from batches.
    Args:
    - batch (list): List of samples from the dataset.

    Returns:
    - tuple: Collated batch
    """
    filtered_batch = [(img, lbl, attr) for img, lbl, attr in batch if img is not None and lbl is not None and attr is not None]
    if not filtered_batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def extract_embeddings(dataloader, model, use_gpu):
    """
    Extract embeddings from the dataloader using the trained autoencoder model.

    Parameters:
    - dataloader (DataLoader): DataLoader for the dataset.
    - model (nn.Module): Trained autoencoder model.
    - use_gpu (bool): Flag to determine whether to use GPU for training.

    Returns:
    - embeddings (torch.Tensor): Extracted embeddings.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _, _ in dataloader:
            if use_gpu:
                images = images.cuda()
            outputs = model.encoder(images)
            embeddings.append(outputs.cpu())
    embeddings = torch.cat(embeddings)
    # Flatten embeddings to 2D
    embeddings = embeddings.view(embeddings.size(0), -1)
    return embeddings
