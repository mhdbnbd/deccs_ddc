import os
import logging
import argparse
import json
from dataset import AwA2Dataset
from model import Autoencoder
from train import train_autoencoder
from utils import extract_embeddings, create_sample_dataset, generate_labels_file, custom_collate
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import setup_logging

setup_logging()

def save_detailed_results(output_path, image_paths, clusters, embeddings, labels, losses, epochs):
    """
    Saves detailed results to a JSON file, including embeddings, clusters, labels.

    Args:
    - output_path (str): Path to save the results.
    - image_paths (list): List of image paths.
    - clusters (list): Cluster assignments for each image.
    - embeddings (list): Embeddings for each image.
    - labels (list): Labels for each image.
    - losses (list): Training losses per epoch.
    - epochs (int): Number of epochs.
    """
    results = []
    for i in range(len(image_paths)):
        result = {
            'image_path': image_paths[i],
            'cluster': int(clusters[i]),
            'embedding': embeddings[i].tolist(),
            'label': int(labels[i])
        }
        results.append(result)

    output = {
        'epochs': epochs,
        'training_losses': losses,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    logging.info(f"Results saved to {output_path}")

def main(use_gpu, use_sample):
    """
    Main function for dataset processing, model training, and clustering.
    """
    source_dir = "data/Animals_with_Attributes2"
    dataset_dir = "AwA2-data-sample"
    pred_file = "data/Animals_with_Attributes2/predicate-matrix-continuous.txt"

    if use_sample:
        create_sample_dataset(source_dir, dataset_dir, sample_size=50)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info("Image transformations defined")

    # Check if the labels file is empty
    if os.stat(attr_file).st_size == 0:
        logging.error("The labels file is empty. Ensure that the labels file is generated correctly and contains data.")
        return

    # Create dataset and dataloader
    logging.info("Creating dataset and dataloader")
    try:
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform)
        dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        logging.info(f"Dataset created with {len(awa2_dataset)} samples.")
        logging.info(f"Dataloader created with {len(dataloader)} batches.")
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    # Initialize the autoencoder
    logging.info("Initializing the autoencoder")
    autoencoder = Autoencoder()

    # Training
    losses = []
    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = train_autoencoder(dataloader, autoencoder, use_gpu)
        losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} completed with loss: {epoch_loss}")

    # Save the trained autoencoder model
    model_save_path = "autoencoder_main.pth"
    torch.save(autoencoder.state_dict(), model_save_path)
    logging.info(f"Trained autoencoder model saved at {model_save_path}")

    # Extract embeddings
    logging.info("Extracting embeddings using the trained autoencoder")
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)
    logging.info(f"Extracted embeddings with shape {embeddings.shape}")

    # Reshape embeddings for KMeans
    embeddings = embeddings.view(embeddings.size(0), -1)

    # Apply KMeans clustering on embeddings
    logging.info("Applying KMeans clustering on embeddings")
    n_clusters = 5  # Adjust based on your need
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())
    logging.info(f"Clustering completed. Clusters: {clusters}")

    # Save detailed results
    output_results_path = "detailed_results_main.json"
    save_detailed_results(output_results_path,
                          awa2_dataset.image_paths,
                          clusters,
                          embeddings.cpu().detach().numpy(),
                          awa2_dataset.labels,
                          losses,
                          num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
