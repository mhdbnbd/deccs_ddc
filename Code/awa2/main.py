"""
creating a sample dataset,
training an autoencoder model, extracting embeddings, and performing
KMeans clustering on the embeddings.

Functions:
    generate_labels_file(img_dir, labels_file): Generates a labels file for the dataset.
    create_sample_dataset(source_dir, target_dir, sample_size): Creates a sample dataset from the source dataset.
    custom_collate(batch): Custom collate function to filter out None values from batches.
    main(use_gpu, use_sample): Main function to orchestrate the dataset processing, model training, and clustering.
"""

import os
import logging
import shutil
import random
import argparse
import pandas as pd
import json
from dataset import AwA2Dataset
from model import Autoencoder
from train import train_autoencoder
from utils import extract_embeddings
from sklearn.cluster import KMeans
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_labels_file(img_dir, labels_file):
    """
    Generate a labels file for the dataset.

    Parameters:
    img_dir (str): Directory containing the image folders.
    labels_file (str): Path to the labels file to be created.

    This function traverses the image directory, assigns a class label to each image
    based on the directory it is found in, and writes this information to the labels file.
    """
    with open(labels_file, 'w') as f:
        for class_label, class_dir in enumerate(os.listdir(img_dir)):
            class_path = os.path.join(img_dir, class_dir)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    if img_filename.endswith('.jpg'):
                        f.write(f"{os.path.join(class_dir, img_filename)} {class_label}\n")
    logging.info(f"Labels file created at {labels_file}")

def create_sample_dataset(source_dir, target_dir, sample_size=100):
    """
    Create a sample dataset from the source dataset.

    Parameters:
    source_dir (str): Directory containing the source dataset.
    target_dir (str): Directory to save the sample dataset.
    sample_size (int): Number of images to include in the sample dataset.

    This function randomly selects a subset of images from the source dataset,
    copies them to the target directory, and creates a corresponding labels file.
    """
    # Clear target directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    img_dir = os.path.join(source_dir, "JPEGImages")
    labels_file = os.path.join(source_dir, "AwA2-labels.txt")
    sample_img_dir = os.path.join(target_dir, "JPEGImages")
    sample_labels_file = os.path.join(target_dir, "AwA2-labels.txt")

    if not os.path.exists(sample_img_dir):
        os.makedirs(sample_img_dir, exist_ok=True)

    # Generate labels file if it doesn't exist
    if not os.path.exists(labels_file):
        logging.warning(f"Labels file {labels_file} not found. Generating a new one.")
        generate_labels_file(img_dir, labels_file)

    # Get a list of all images (recursively)
    all_images = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith('.jpg'):
                all_images.append(os.path.join(root, file))

    # Check if sample_size is larger than the number of available images
    if sample_size > len(all_images):
        sample_size = len(all_images)
        logging.warning(f"Sample size adjusted to {sample_size} due to limited number of images.")

    sampled_images = random.sample(all_images, sample_size)

    # Copy sampled images
    for img in sampled_images:
        target_path = os.path.join(sample_img_dir, os.path.relpath(img, img_dir))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Remove target file if it exists to handle read-only files
        if os.path.exists(target_path):
            os.remove(target_path)

        shutil.copy(img, target_path)
        logging.debug(f"Copied {img} to {target_path}")

    # Create labels file for sampled images
    with open(labels_file, "r") as f:
        lines = f.readlines()

    sampled_lines = [line for line in lines if os.path.relpath(line.split()[0], '').replace('\\', '/') in [os.path.relpath(img, img_dir).replace('\\', '/') for img in sampled_images]]

    with open(sample_labels_file, "w") as f:
        f.writelines(sampled_lines)

    logging.info(f"Sample of {sample_size} images created in {sample_img_dir}")
    logging.info(f"Sample labels file created at {sample_labels_file}")

def custom_collate(batch):
    """
    Custom collate function to filter out None values from batches.

    Parameters:
    batch (list): List of samples from the dataset.

    Returns:
    Filtered batch with None values removed.
    """
    filtered_batch = [item for item in batch if item[0] is not None]
    if len(filtered_batch) < len(batch):
        logging.warning(f"Filtered out {len(batch) - len(filtered_batch)} samples containing None values")
    if len(filtered_batch) > 0:
        return torch.utils.data.default_collate(filtered_batch)
    else:
        logging.error("All samples in the batch are None. Returning empty batch.")
        return torch.utils.data.default_collate([])

def main(use_gpu, use_sample):
    """
    Parameters:
    use_gpu (bool): Flag to indicate whether to use GPU if available.
    use_sample (bool): Flag to indicate whether to use a sample dataset or the full dataset.

    This function creates a sample dataset if specified, defines image transformations,
    creates the dataset and dataloader, initializes and trains the autoencoder,
    extracts embeddings, and performs KMeans clustering on the embeddings.
    """
    # Directory of the existing AwA2 dataset
    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    # Directory to save the sample dataset
    dataset_dir = "AwA2-data-sample"

    if use_sample:
        # Create a sample dataset
        create_sample_dataset(source_dir, dataset_dir, sample_size=100)
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
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, transform=transform)
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
    n_clusters = 5  # Adjust this based on your specific requirements
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())
    logging.info(f"Clustering completed. Clusters: {clusters}")

    # Save the clustering results in a pretty format
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