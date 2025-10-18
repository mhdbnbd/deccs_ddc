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
from model import ConstrainedAutoencoder
from train import train_constrained_autoencoder
from utils import extract_embeddings, create_sample_dataset, custom_collate, setup_logging, generate_notebook, save_detailed_results
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np

# Setup logging for the script
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
    source_dir = "data/Animals_with_Attributes2"
    dataset_dir = "AwA2-data-sample-tags2"
    pred_file = "data/Animals_with_Attributes2/predicate-matrix-continuous.txt"

    if use_sample:
        create_sample_dataset(source_dir, dataset_dir, sample_size=100)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    try:
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform)
        dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        logging.info(f"Dataset created with {len(awa2_dataset)} samples.")
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    # Initialize and train the constrained autoencoder
    autoencoder = ConstrainedAutoencoder()

    training_losses = []
    num_epochs = 4

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = train_constrained_autoencoder(dataloader, autoencoder, use_gpu)
        training_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} completed with loss: {epoch_loss}")

    # Save the trained constrained autoencoder model
    model_save_path = "autoencoder_tags2.pth"
    torch.save(autoencoder.state_dict(), model_save_path)
    logging.info(f"Trained constrained autoencoder model saved at {model_save_path}")

    # Extract embeddings using the trained autoencoder
    embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)

    # Use symbolic tags as additional features (concatenate with embeddings)
    symbolic_tags = torch.tensor(awa2_dataset.symbolic_tags)
    combined_features = torch.cat((embeddings, symbolic_tags), dim=1)

    # Standardize features before clustering
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features.cpu().detach().numpy())

    # Apply KMeans clustering on combined features
    n_clusters = len(set(awa2_dataset.labels))  
    logging.info(f"Applying KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(combined_features)

    # Calculate final accuracy and ARI
    #add NMI
    true_labels = awa2_dataset.labels
    acc = calculate_clustering_accuracy(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)
    logging.info(f"Final clustering accuracy (ACC): {acc}")
    logging.info(f"Adjusted Rand Index (ARI): {ari}")

    # Save detailed results
    output_results_path = "detailed_results_tags2.json"
    save_detailed_results(output_results_path,
                          awa2_dataset.image_paths,
                          clusters,
                          embeddings.cpu().detach().numpy(),
                          awa2_dataset.labels,
                          symbolic_tags=awa2_dataset.symbolic_tags,
                          losses=training_losses,
                          accuracy=acc,
                          epochs=num_epochs)

    # Generate results notebook
    output_notebook_path = "results_notebook_tags2.ipynb"
    generate_notebook(output_results_path, output_notebook_path)
    logging.info(f"Notebook generated at {output_notebook_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing with symbolic tags.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
