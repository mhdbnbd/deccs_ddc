import json
import logging
import logging
import nbformat as nbf
import nbformat as nbf
import os
import random
import shutil
import torch


def setup_logging(log_filename='maintag2_sampled.log'):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define a common formatter and add it to both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def extract_embeddings(dataloader, model, use_gpu):
    """
    Extract embeddings using the trained autoencoder model.
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            logging.debug(f"Batch {batch_idx} processed with image shape: {images.shape}")

            # Forward pass through autoencoder encoder
            encoded = model.encoder(images)
            logging.debug(f"Encoded embeddings for batch {batch_idx}: {encoded.shape}")
            encoded = torch.nn.functional.adaptive_avg_pool2d(encoded, 1)
            embeddings.append(encoded.view(encoded.size(0), -1).cpu())

    embeddings = torch.cat(embeddings, dim=0)
    logging.info(f"Total extracted embeddings shape: {embeddings.shape}")
    return embeddings


def create_sample_dataset(source_dir, target_dir, classes_file, sample_size=100):
    """
    Create a sample dataset, ensuring correct label assignments from `classes.txt`.

    Args:
    - source_dir (str): Path to full dataset.
    - target_dir (str): Path to sample dataset.
    - classes_file (str): Path to `classes.txt` to enforce correct labels.
    - sample_size (int): Number of images to sample.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    img_dir = os.path.join(source_dir, "JPEGImages")
    labels_file = os.path.join(source_dir, "AwA2-labels.txt")
    sample_img_dir = os.path.join(target_dir, "JPEGImages")
    sample_labels_file = os.path.join(target_dir, "AwA2-labels.txt")

    if not os.path.exists(sample_img_dir):
        os.makedirs(sample_img_dir, exist_ok=True)

    # Read correct class mapping from `classes.txt`
    class_mapping = {}
    with open(classes_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                class_id = int(parts[0])  # Keep `classes.txt` order
                class_name = parts[1]
                class_mapping[class_name] = class_id

    # Load images and labels together
    image_label_pairs = []
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_folder = parts[0].split('/')[0]  # Extract class name
            correct_label = class_mapping.get(class_folder, -1)
            if correct_label != -1:
                image_label_pairs.append((parts[0], correct_label))

    if sample_size > len(image_label_pairs):
        sample_size = len(image_label_pairs)
        logging.warning(f"Sample size adjusted to {sample_size} due to limited number of images.")

    # Sample while maintaining label consistency
    sampled_pairs = random.sample(image_label_pairs, sample_size)

    # Copy images and save sampled labels
    with open(sample_labels_file, "w") as f:
        for img_rel_path, label in sampled_pairs:
            source_path = os.path.join(img_dir, img_rel_path)
            target_path = os.path.join(sample_img_dir, img_rel_path)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(source_path, target_path)

            f.write(f"{img_rel_path} {label}\n")

    logging.info(f"Sample of {sample_size} images created in {sample_img_dir}")
    logging.info(f"Sample labels file created at {sample_labels_file}")

def generate_labels_file(img_dir, labels_file, classes_file):
    """
    Generate a labels file (`AwA2-labels.txt`) ensuring labels match `classes.txt` order.

    Args:
    - img_dir (str): Path to directory containing image class folders.
    - labels_file (str): Output labels file.
    - classes_file (str): Path to `classes.txt` to enforce correct label mapping.
    """
    # Read `classes.txt` and create a mapping: class_name → correct_label
    class_mapping = {}
    with open(classes_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                class_id = int(parts[0])  # Keep 1-based indexing from `classes.txt`
                class_name = parts[1]  # Class folder name
                class_mapping[class_name] = class_id  # Enforce label consistency

    os.makedirs(os.path.dirname(labels_file), exist_ok=True)

    with open(labels_file, 'w') as f:
        for class_name, class_id in class_mapping.items():
            class_path = os.path.join(img_dir, class_name)
            if os.path.isdir(class_path):
                for img_filename in os.listdir(class_path):
                    if img_filename.endswith('.jpg'):
                        f.write(f"{os.path.join(class_name, img_filename)} {class_id}\n")

    logging.info(f"Labels file created at {labels_file}, using official `classes.txt` mapping.")

def custom_collate(batch):
    # Filter out None samples
    batch = [b for b in batch if b[0] is not None]
    
    if len(batch) == 0:
        logging.warning("All samples in the batch are None. Skipping this batch.")
        return []  # Instead of raising StopIteration, return an empty batch

    return torch.utils.data.default_collate(batch)

def save_detailed_results(output_path, image_paths, clusters, embeddings, labels, symbolic_tags=None, losses=None, accuracy=None, epochs=None):
    """
    Saves detailed results to a JSON file, including embeddings, clusters, labels, and tags.

    Args:
    - output_path (str): Path to save the results.
    - image_paths (list): List of image paths.
    - clusters (list): Cluster assignments for each image.
    - embeddings (list): Embeddings for each image.
    - labels (list): Labels for each image.
    - symbolic_tags (list or None): Symbolic tags for each image. Can be None if not available.
    - losses (list or None): Training losses per epoch. Can be None if not applicable.
    - accuracy (float or None): Final accuracy after clustering. Can be None if not available.
    - epochs (int or None): Number of epochs. Can be None if not applicable.
    """
    results = []
    for i in range(len(image_paths)):
        result = {
            'image_path': image_paths[i],
            'cluster': int(clusters[i]),
            'embedding': embeddings[i].tolist(),
            'label': int(labels[i])
        }
        # Add symbolic tag information if provided
        if symbolic_tags is not None:
            result['symbolic_tag'] = symbolic_tags[i].tolist()
        results.append(result)
    
    output = {
        'epochs': epochs if epochs is not None else "Not provided",
        'training_losses': losses if losses is not None else "Not provided",
        'final_accuracy': accuracy if accuracy is not None else "Not provided",
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    logging.info(f"Results saved to {output_path}")

def generate_notebook(results_file, output_notebook):
    """
    Generates a Jupyter notebook to present the results and the steps taken to achieve them.
    
    Args:
    - results_file (str): Path to the JSON file containing the results (e.g., clustering results, embeddings, etc.).
    - output_notebook (str): Path to save the generated notebook.
    """
    logging.info(f"Generating notebook at {output_notebook}")
    
    nb = nbf.v4.new_notebook()

    # 1. Add introduction markdown cell
    intro_text = """# Results Notebook

This notebook presents the results of the clustering process performed on the AwA2 dataset using autoencoders and KMeans clustering.

## Table of Contents:
1. Data Loading
2. Model Training
3. Clustering and Embeddings
4. Visualization of Results
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(intro_text))

    # 2. Add a code cell for loading data
    data_loading_code = f"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results file
results_file = '{results_file}'
with open(results_file, 'r') as f:
    results = json.load(f)

print("Results loaded successfully")

# Extract clusters and embeddings from results
clusters = [result['cluster'] for result in results['results']]
embeddings = np.array([result['embedding'] for result in results['results']])
image_paths = [result['image_path'] for result in results['results']]

print("Clusters and embeddings extracted.")
"""
    nb['cells'].append(nbf.v4.new_code_cell(data_loading_code))

    # 3. Add a code cell for plotting loss curves
    plot_loss_code = """
# Plot the training loss over epochs
epochs = results['epochs']
losses = results['training_losses']

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
"""
    nb['cells'].append(nbf.v4.new_code_cell(plot_loss_code))

    # 4. Add a code cell for clustering results visualization
    cluster_vis_code = """
# Visualize the clustering results
# Count how many samples per cluster
unique_clusters, counts = np.unique(clusters, return_counts=True)

plt.figure(figsize=(8, 5))
plt.bar(unique_clusters, counts, color='skyblue')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.show()
"""
    nb['cells'].append(nbf.v4.new_code_cell(cluster_vis_code))

    # 5. Add embedding visualization (e.g., PCA or t-SNE)
    embed_vis_code = """
from sklearn.decomposition import PCA

# Reduce dimensionality of embeddings with PCA for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', s=30)
plt.colorbar()
plt.title('Embeddings Visualized using PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
"""
    nb['cells'].append(nbf.v4.new_code_cell(embed_vis_code))

    # 6. Add a summary markdown cell
    summary_text = """
## Summary

- We trained an autoencoder on the AwA2 dataset and extracted embeddings.
- The embeddings were clustered using KMeans with the number of clusters set to 50.
- Loss per epoch was tracked, and the cluster distribution and embeddings were visualized.
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(summary_text))

    # Save the notebook
    with open(output_notebook, 'w') as f:
        nbf.write(nb, f)

    logging.info(f"Notebook saved to {output_notebook}")

