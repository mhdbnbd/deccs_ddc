import os
import logging
import argparse
import json
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import AwA2Dataset
from model import Autoencoder
from train import train_autoencoder
from utils import extract_embeddings, create_sample_dataset, custom_collate


def main(use_gpu, use_sample):
    # Setup paths and create sample dataset if needed
    source_dir = "data/Animals_with_Attributes2"
    dataset_dir = "AwA2-data-sample"
    pred_file = "data/Animals_with_Attributes2/predicate-matrix-continuous.txt"

    if use_sample:
        create_sample_dataset(source_dir, dataset_dir, sample_size=1000)
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

    # Create the dataset and dataloader
    awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform)
    dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # Initialize and train the autoencoder
    autoencoder = Autoencoder()
    train_autoencoder(dataloader, autoencoder, use_gpu)

    # Extract embeddings using the trained autoencoder
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)

    # Concatenate embeddings with symbolic tags
    symbolic_tags = torch.tensor(awa2_dataset.symbolic_tags)
    combined_features = torch.cat((embeddings, symbolic_tags), dim=1)

    # Standardize features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features.cpu().detach().numpy())

    # Apply KMeans clustering on combined features
    n_clusters = 5  # to be adjusted
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(combined_features)

    # Save the clustering results
    results = {f"Image_{i}": int(cluster) for i, cluster in enumerate(clusters)}
    with open("clustering_results_tags.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Clustering results saved to clustering_results_tags.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
