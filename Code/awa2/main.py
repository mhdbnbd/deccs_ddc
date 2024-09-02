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

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main(use_gpu, use_sample):
    """
    Main function to orchestrate the dataset processing, model training, and clustering.

    Parameters:
    use_gpu (bool): Flag to indicate whether to use GPU if available.
    use_sample (bool): Flag to indicate whether to use a sample dataset or the full dataset.
    """
    # Directory of the existing AwA2 dataset
    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    # Directory to save the sample dataset
    dataset_dir = "AwA2-data-sample"

    pred_file = "data/AwA2-data/Animals_with_Attributes2/predicate-matrix-continuous.txt"

    if use_sample:
        # Create a sample dataset
        create_sample_dataset(source_dir, dataset_dir, sample_size=15)
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
        logging.info(f"Dataset and dataloader created with {len(awa2_dataset)} samples")
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    # Initialize the autoencoder
    logging.info("Initializing the autoencoder")
    autoencoder = Autoencoder()

    # Train the autoencoder
    logging.info("Training the autoencoder")
    train_autoencoder(dataloader, autoencoder, use_gpu)

    # Extract embeddings
    logging.info("Extracting embeddings using the trained autoencoder")
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)
    logging.info(f"Extracted embeddings with shape {embeddings.shape}")

    # Reshape embeddings for KMeans
    embeddings = embeddings.view(embeddings.size(0), -1)

    # Apply KMeans clustering on embeddings
    logging.info("Applying KMeans clustering on embeddings")
    n_clusters = 5  # to be adjusted
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())
    logging.info(f"Clustering completed. Clusters: {clusters}")

    # Save the clustering results
    results = {f"Image_{i}": int(cluster) for i, cluster in enumerate(clusters)}
    with open("clustering_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Clustering results saved to clustering_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)

# python3 main.py --use_gpu --use_sample