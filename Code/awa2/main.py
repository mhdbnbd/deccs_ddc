import os
import logging
import argparse
import json
from dataset import AwA2Dataset
from model import Autoencoder
from train import train_autoencoder
from utils import extract_embeddings, create_sample_dataset, custom_collate, setup_logging
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

setup_logging()

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

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info("Image transformations defined")

    # Check if labels file exists
    if os.stat(attr_file).st_size == 0:
        logging.error("The labels file is empty.")
        return

    logging.info("Creating dataset and dataloader")
    try:
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform)
        dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    if len(dataloader) == 0:
        logging.error("No valid samples in DataLoader.")
        return

    # Initialize autoencoder
    logging.info("Initializing the autoencoder")
    autoencoder = Autoencoder()

    # Train the autoencoder
    logging.info("Training the autoencoder")
    train_autoencoder(dataloader, autoencoder, use_gpu)

    # Save the trained autoencoder model
    model_save_path = "autoencoder_main.pth"
    torch.save(autoencoder.state_dict(), model_save_path)
    logging.info(f"Trained autoencoder model saved at {model_save_path}")

    # Extract embeddings
    logging.info("Extracting embeddings")
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)

    # Reshape embeddings for clustering
    embeddings = embeddings.view(embeddings.size(0), -1)

    # Apply KMeans clustering
    logging.info("Applying KMeans clustering")
    n_clusters = 5  # Adjust if necessary
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())

    # Save clustering results
    results = {awa2_dataset.image_paths[i]: int(cluster) for i, cluster in enumerate(clusters)}
    with open("clustering_results_main.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Clustering results saved to clustering_results_main.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
