import torch

def extract_embeddings(dataloader, model, use_gpu):
    """
    Extract embeddings using the trained autoencoder model.

    Parameters:
    dataloader (DataLoader): Dataloader for the data.
    model (nn.Module): Trained autoencoder model.
    use_gpu (bool): Flag to indicate whether to use GPU if available.

    Returns:
    torch.Tensor: Extracted embeddings.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for images, _, _, _ in dataloader:
            images = images.to(device)
            encoded = model.encoder(images)
            embeddings.append(encoded.view(encoded.size(0), -1).cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings
