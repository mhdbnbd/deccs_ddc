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
from train import train_constrained_autoencoder
from utils import extract_embeddings, create_sample_dataset, custom_collate, setup_logging

# Setup logging for the script
setup_logging()

def save_detailed_results(output_path, image_paths, clusters, embeddings, labels, symbolic_tags, losses, accuracy, epochs):
    """
    Saves detailed results to a JSON file, including embeddings, clusters, labels, and tags.

    Args:
    - output_path (str): Path to save the results.
    - image_paths (list): List of image paths.
    - clusters (list): Cluster assignments for each image.
    - embeddings (list): Embeddings for each image.
    - labels (list): Labels for each image.
    - symbolic_tags (list): Symbolic tags for each image.
    - losses (list): Training losses per epoch.
    - accuracy (float): Final accuracy after clustering.
    - epochs (int): Number of epochs.
    """
    results = []
    for i in range(len(image_paths)):
        result = {
            'image_path': image_paths[i],
            'cluster': int(clusters[i]),
            'embedding': embeddings[i].tolist(),
            'label': int(labels[i]),
            'symbolic_tag': symbolic_tags[i].tolist()
        }
        results.append(result)
    
    output = {
        'epochs': epochs,
        'training_losses': losses,
        'final_accuracy': accuracy,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    logging.info(f"Results saved to {output_path}")

def main(use_gpu, use_sample):
    source_dir = "data/Animals_with_Attributes2"
    dataset_dir = "AwA2-data-sample"
    pred_file = "data/Animals_with_Attributes2/predicate-matrix-continuous.txt"

    # Use either the full dataset or a sample dataset
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

    try:
        # Create dataset and dataloader
        awa2_dataset = AwA2Dataset(img_dir=img_dir, attr_file=attr_file, pred_file=pred_file, transform=transform)
        dataloader = DataLoader(awa2_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
        logging.info(f"Dataset created with {len(awa2_dataset)} samples.")
        logging.info(f"Dataloader created with {len(dataloader)} batches.")
    except Exception as e:
        logging.error(f"Error creating dataset and dataloader: {e}")
        return

    # Initialize and train the constrained autoencoder
    logging.info("Initializing and training the constrained autoencoder")
    autoencoder = Autoencoder()

    # Store training losses
    losses = []
    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = train_constrained_autoencoder(dataloader, autoencoder, use_gpu)
        losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} completed with loss: {epoch_loss}")

    # Save the trained constrained autoencoder model
    model_save_path = "autoencoder_tags2.pth"
    torch.save(autoencoder.state_dict(), model_save_path)
    logging.info(f"Trained constrained autoencoder model saved at {model_save_path}")

    # Extract embeddings using the trained autoencoder
    logging.info("Extracting embeddings from the constrained autoencoder")
    autoencoder_embeddings = extract_embeddings(dataloader, autoencoder, use_gpu)

    # Use symbolic tags as additional features (concatenate with embeddings)
    symbolic_tags = torch.tensor(awa2_dataset.symbolic_tags)
    combined_features = torch.cat((autoencoder_embeddings, symbolic_tags), dim=1)
    logging.info(f"Combined features shape: {combined_features.shape}")

    # Standardize the combined features
    logging.info("Standardizing combined features")
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features.cpu().detach().numpy())

    # Apply KMeans clustering on combined features
    n_clusters = 5  # Can be tuned or adjusted dynamically
    logging.info(f"Applying KMeans with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(combined_features)

    # Calculate final accuracy (you can adjust this to suit your needs)
    correct = sum(1 for i in range(len(clusters)) if clusters[i] == awa2_dataset.labels[i])
    accuracy = correct / len(clusters)
    logging.info(f"Final clustering accuracy: {accuracy}")

    # Save the clustering results with detailed information
    output_results_path = "detailed_results_tags2.json"
    save_detailed_results(output_results_path,
                          awa2_dataset.image_paths,
                          clusters,
                          autoencoder_embeddings.cpu().detach().numpy(),
                          awa2_dataset.labels,
                          awa2_dataset.symbolic_tags,
                          losses,
                          accuracy,
                          num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AwA2 dataset processing with symbolic tags.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--use_sample', action='store_true', help='Use sample dataset instead of full dataset.')
    args = parser.parse_args()
    main(args.use_gpu, args.use_sample)
