import os
import logging
import argparse
import json
from dataset import AwA2Dataset
from model import Autoencoder
from train import train_autoencoder, evaluate_autoencoder
from utils import extract_embeddings, create_sample_dataset, custom_collate, setup_logging, generate_notebook, save_detailed_results
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

setup_logging()

def calculate_clustering_accuracy(true_labels, predicted_clusters):
    """
    Compute clustering accuracy (ACC) by finding the best one-to-one mapping between clusters and true labels.
    """
    max_label = max(true_labels) + 1  # Determine the number of unique true labels
    max_cluster = max(predicted_clusters) + 1  # Determine the number of unique predicted clusters

    # Initialize the contingency matrix with dynamic shape
    contingency_matrix = np.zeros((max_label, max_cluster))

    # Populate the contingency matrix
    for i, (true_label, cluster) in enumerate(zip(true_labels, predicted_clusters)):
        contingency_matrix[true_label, cluster] += 1

    # Use the Hungarian algorithm to find the best cluster-label mapping
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    best_mapping = contingency_matrix[row_ind, col_ind].sum()

    acc = best_mapping / len(true_labels)  # Clustering accuracy
    return acc

def main(use_gpu, use_sample):
    """
    Main function for dataset processing, model training, and clustering.
    """
    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-data-sample"
    pred_file = "data/AwA2-data/Animals_with_Attributes2/predicate-matrix-continuous.txt"
    classes_file = "data/AwA2-data/Animals_with_Attributes2/classes.txt"

    if use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=100)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    logging.info("Image transformations defined")

    if os.stat(attr_file).st_size == 0:
        logging.error("The labels file is empty. Ensure that the labels file is generated correctly and contains data.")
        return

    logging.info("Creating dataset and dataloader")
    try:
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform,
                                   classes_file=classes_file, train=True)
        test_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform,
                                   classes_file=classes_file, train=False)

        dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        logging.info(f"Dataset created with {len(awa2_dataset)} samples.")
        logging.info(f"Dataloader created with {len(dataloader)} batches.")
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    # Initialize the autoencoder
    autoencoder = Autoencoder()

    # Training
    training_losses = []
    num_epochs = 6

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train_autoencoder(dataloader, autoencoder, use_gpu)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Evaluate after every epoch
        test_loss = evaluate_autoencoder(test_loader, autoencoder, use_gpu)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}")

    # Save the trained autoencoder model
    model_save_path = "autoencoder_main.pth"
    torch.save(autoencoder.state_dict(), model_save_path)
    logging.info(f"Trained autoencoder model saved at {model_save_path}")

    # Extract embeddings
    logging.info("Extracting embeddings using the trained autoencoder")
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)
    logging.info(f"Extracted embeddings with shape {embeddings.shape}")

    embeddings = embeddings.view(embeddings.size(0), -1)

    # Apply KMeans clustering on embeddings
    n_clusters = len(set(awa2_dataset.labels))
    logging.info(f"Applying KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings.cpu().detach().numpy())

    # Calculate final accuracy and ARI
    true_labels = awa2_dataset.labels
    acc = calculate_clustering_accuracy(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)
    logging.info(f"Final clustering accuracy (ACC): {acc}")
    logging.info(f"Adjusted Rand Index (ARI): {ari}")

    # Save detailed results
    output_results_path = "detailed_results_main.json"
    save_detailed_results(output_results_path,
                          awa2_dataset.image_paths,
                          clusters,
                          embeddings.cpu().detach().numpy(),
                          awa2_dataset.labels,
                          losses=training_losses,
                          accuracy=acc,
                          epochs=num_epochs)

    # Generate results notebook
    output_notebook_path = "results_notebook_main.ipynb"
    generate_notebook(output_results_path, output_notebook_path)
    logging.info(f"Notebook generated at {output_notebook_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
