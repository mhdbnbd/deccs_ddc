import inspect
import json
import logging
import os
import random
import shutil
import time


from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from clustpy.deep import DEC, DDC
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, \
    silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def setup_logging(log_filename='maintag2_sampled.log'):
    logger = logging.getLogger()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(log_filename, maxBytes=5e6, backupCount=3)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def extract_embeddings(dataloader, model, use_gpu):
    """
    Extract embeddings using the trained autoencoder model.
    Works whether dataset returns (img, tag) or (img, tag, idx).
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            # Handle both 2-tuple (img, tag) and 3-tuple (img, tag, idx)
            if len(batch) == 3:
                images, _, _ = batch
            elif len(batch) == 2:
                images, _ = batch
            else:
                raise ValueError(f"Unexpected batch structure: {len(batch)} elements")

            torch.backends.cudnn.benchmark = True
            images = images.to(device, non_blocking=True)
            encoded = model.encoder(images)

            # Global pooling (make embeddings flat)
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
        return None  # Instead of raising StopIteration, return an empty batch
    return torch.utils.data.default_collate(batch)

def evaluate_clustering(embeddings, true_labels, k=None, mode_desc=""):
    """
    Perform consensus clustering on embeddings using multiple clustering algorithms,
    following the DECCS paper setup (KMeans, GMM, Agglomerative, Spectral, DBSCAN),
    and aggregate results via consensus spectral clustering.

    Args:
        embeddings (ndarray): Learned latent features (n_samples x n_features).
        true_labels (array): Ground truth class labels (for evaluation only).
        k (int): Number of clusters (defaults to number of unique labels).
        mode_desc (str): Descriptor for logging.

    Returns:
        dict: Dictionary containing metrics and final cluster assignments.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    assert isinstance(embeddings, np.ndarray), "Embeddings must be a NumPy array or torch tensor"

    if k is None:
        k = len(np.unique(true_labels))

    logging.info(f"Running DECCS-style ensemble clustering with {k} clusters...")

    # Normalize embeddings
    X = StandardScaler().fit_transform(embeddings)

    # Define base clusterers (matching DECCS paper setup)
    clusterers = {
        "KMeans": KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            algorithm="elkan",
            random_state=42
        ),
        "Spectral": SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",     # avoids dense kernel matrix
            n_neighbors=15,                  # build sparse graph instead of full
            assign_labels="kmeans",
            eigen_solver="arpack",            # stable small-memory solver
            random_state=42
        ),
        "GMM": GaussianMixture(n_components=k, random_state=42,
                               covariance_type="full", reg_covar=1e-4, max_iter=200),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",          # Euclidean-space linkage
            compute_full_tree=False  # avoids full dendrogram computation
        ),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }

    # Run base clusterings
    base_labels = []
    for name, algo in clusterers.items():
        try:
            labels = algo.fit_predict(X)
            base_labels.append(labels)
            logging.info(f"Base clustering '{name}' completed.")
        except Exception as e:
            logging.warning(f"Base clustering '{name}' failed: {e}")

    base_labels = np.array(base_labels)
    logging.info(f"Completed {len(base_labels)} base clusterings.")

    # === Build consensus matrix & cluster it ===
    consensus_matrix = build_consensus_matrix(base_labels)
    consensus_matrix /= consensus_matrix.max()
    final_labels = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    ).fit_predict(consensus_matrix)

    # === Compute metrics ===
    acc = clustering_acc(true_labels, final_labels)
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)

    try:
        sil = silhouette_score(X, final_labels)
    except Exception as e:
        logging.warning(f"Silhouette computation failed: {e}")
        sil = float('nan')

    logging.info(
        f"[{mode_desc}] ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil:.4f}"
    )

    with open("results_cluster_metrics.json", "w") as f:
        json.dump({"acc": acc, "ari": ari, "nmi": nmi, "silhouette": sil}, f, indent=2)

    return {
        "acc": float(acc),
        "ari": float(ari),
        "nmi": float(nmi),
        "silhouette": float(sil),
        "clusters": final_labels.tolist()
    }

def get_base_clusterings(embeddings_np, n_clusters=10):
    """
    Build a hybrid ensemble of deep + classical clusterers for DECCS.

    Deep:  DEC, DDC  (from ClustPy)
    Classical: KMeans, Spectral, GMM, Agglomerative, DBSCAN  (from scikit-learn)

    This version automatically adapts to ClustPy API differences (v0.0.2+)
    and produces a robust ensemble for consensus matrix construction.
    """

    # --- Normalize embeddings ---
    X = StandardScaler().fit_transform(embeddings_np)

    # --- Prepare ensemble containers ---
    base_labels = []
    deep_models = {"DEC": DEC, "DDC": DDC}
    classical_models = {
        "KMeans": KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            algorithm="elkan",
            random_state=42
        ),
        "Spectral": SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",     # avoids dense kernel matrix
            n_neighbors=15,                  # build sparse graph instead of full
            assign_labels="kmeans",
            eigen_solver="arpack",            # stable small-memory solver
            random_state=42
        ),
        "GMM": GaussianMixture(n_components=n_clusters, random_state=42,
                               covariance_type="full", reg_covar=1e-4, max_iter=200),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",          # Euclidean-space linkage
            compute_full_tree=False  # avoids full dendrogram computation
        ),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }

    # --- Deep clustering ensemble ---
    for name, Cls in deep_models.items():
        start = time.time()
        try:
            model = Cls(n_clusters=n_clusters) if "n_clusters" in inspect.signature(Cls).parameters else Cls()
            labels = model.fit_predict(X)
            if np.isnan(labels).any():
                logging.warning(f"{name} produced NaNs — skipping.")
                continue
            base_labels.append(labels)
            logging.info(f"[DECCS] Deep base clustering '{name}' completed in {time.time() - start:.2f}s.")
        except Exception as e:
            logging.warning(f"[DECCS] Deep clustering '{name}' failed: {e}")

    # --- Classical clustering ensemble ---
    for name, algo in classical_models.items():
        start = time.time()
        try:
            labels = algo.fit_predict(X)
            base_labels.append(labels)
            logging.info(f"[DECCS] Classical base clustering '{name}' completed.")
            logging.info(f"Base clustering '{name}' completed in {time.time() - start:.2f}s")
        except Exception as e:
            logging.warning(f"[DECCS] Classical clustering '{name}' failed: {e}")


    # --- Combine results ---
    base_labels = np.array(base_labels)
    if base_labels.size == 0:
        raise RuntimeError("All base clusterers failed.")

    logging.info(f"[DECCS] {len(base_labels)} base clusterings successful. Shape={base_labels.shape}")
    return base_labels

def build_consensus_matrix(base_labels):
    """Compute NxN co-association matrix across base clusterings."""
    n_clusterers, n_samples = base_labels.shape
    consensus = np.zeros((n_samples, n_samples), dtype=np.float32)
    for labels in base_labels:
        consensus += (labels[:, None] == labels[None, :]).astype(np.float32)
    return consensus / n_clusterers

def consensus_consistency_loss(embeddings, consensus_matrix_np):
    """Encourage embedding cosine similarity to match consensus similarity."""
    device = embeddings.device
    consensus = torch.tensor(consensus_matrix_np, dtype=torch.float32, device=device)
    z = F.normalize(embeddings, dim=1)
    sim = torch.mm(z, z.T)
    sim = (sim + 1.0) / 2.0
    return F.mse_loss(sim, consensus)


def clustering_acc(y_true, y_pred):
    y_true_u, y_true = np.unique(y_true, return_inverse=True)
    y_pred_u, y_pred = np.unique(y_pred, return_inverse=True)
    D = np.zeros((y_true_u.size, y_pred_u.size), dtype=np.int64)
    for i in range(y_true.size):
        D[y_true[i], y_pred[i]] += 1
    r, c = linear_sum_assignment(D.max() - D)
    return D[r, c].sum() / y_true.size

# Convert all numpy/tensor values to standard Python types
def default_serializer(o):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    elif isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    elif hasattr(o, 'tolist'):
        return o.tolist()
    else:
        return str(o)


def describe_clusters(embeddings, tags, n_clusters=None):
    """
    Generate human-interpretable cluster descriptions based on tag vectors.

    Args:
        embeddings (np.ndarray): Latent features (N x d)
        tags (np.ndarray): Symbolic tag matrix (N x T)
        n_clusters (int or None): Number of clusters (if None, infer from tags)
        method (str): clustering algorithm to use

    Returns:
        cluster_labels: cluster assignment per sample
        cluster_descriptions: list of avg tag vectors per cluster
    """
    # Automatically infer number of clusters from tags
    if n_clusters is None:
        n_clusters = len(np.unique(tags.argmax(axis=1))) if tags.ndim > 1 else np.unique(tags).size

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(embeddings)

    cluster_descriptions = []
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        if np.sum(cluster_mask) == 0:
            cluster_descriptions.append(np.zeros(tags.shape[1]))
            continue
        # Average tag vector for cluster c
        avg_tags = np.mean(tags[cluster_mask], axis=0)
        cluster_descriptions.append(avg_tags)

    return cluster_labels, np.array(cluster_descriptions)


def load_attribute_names(predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt"):
    """
    Load AwA2 attribute names from predicates.txt.
    Returns:
        List[str]: Ordered list of 85 attribute names.
    """
    if not os.path.exists(predicates_path):
        raise FileNotFoundError(f"Predicates file not found at {predicates_path}")

    attribute_names = []
    with open(predicates_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                attribute_names.append(parts[1])
    return attribute_names


def summarize_clusters_with_attributes(
    cluster_labels,
    cluster_descriptions,
    dataset,
    predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt",
    top_k=5,
    output_json="results_cluster_descriptions.json"
):
    """
    Generate human-readable cluster summaries and save them as JSON.
    Each cluster includes top attributes and a few representative images.

    Args:
        cluster_labels (np.ndarray): Cluster label for each sample.
        cluster_descriptions (np.ndarray): Averaged tag vectors per cluster.
        dataset (Dataset): AwA2 dataset instance (with image_paths and labels).
        predicates_path (str): Path to predicates.txt file.
        top_k (int): Number of top attributes to show per cluster.
        output_json (str): Path to save structured results.
    """
    attribute_names = load_attribute_names(predicates_path)
    results = []

    if cluster_descriptions.shape[1] != len(attribute_names):
        logging.warning(f"Attribute count mismatch: {cluster_descriptions.shape[1]} vs {len(attribute_names)}")

    for cluster_id, desc in enumerate(cluster_descriptions):
        top_attrs = np.argsort(desc)[-top_k:][::-1]
        readable_attrs = [attribute_names[j] for j in top_attrs]

        # Get sample images from this cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        sample_images = []
        for idx in random.sample(list(indices), min(3, len(indices))):  # 3 representative samples
            sample_images.append({
                "image_path": dataset.image_paths[idx],
                "label": int(dataset.labels[idx])
            })

        logging.info(f"Cluster {cluster_id:02d}: top attributes -> {', '.join(readable_attrs)}")
        results.append({
            "cluster_id": int(cluster_id),
            "num_samples": int(len(indices)),
            "top_attributes": readable_attrs,
            "sample_images": sample_images
        })

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=default_serializer)
    logging.info(f"[DDC] Cluster descriptions saved ({len(results)} clusters) → {output_json}")
    return results

def generate_cluster_report(samples_dir="cluster_samples", out="cluster_report"):
    os.makedirs(out, exist_ok=True)
    with open("results_cluster_descriptions.json") as f:
        attrs = json.load(f)
    if isinstance(attrs, list):
        logging.warning("Cluster descriptions loaded as list — converting to dictionary format.")
        attrs = {str(entry["cluster_id"]): entry for entry in attrs}

    for cid, desc in attrs.items():
        canvas = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(canvas)
        txt = f"Cluster {cid}\nTop attributes:\n" + ", ".join(desc["top_attributes"])
        draw.text((10,10), txt, fill="black")
        cluster_dir = os.path.join(samples_dir, f"cluster_{int(cid):02d}")
        x = 10
        for img_name in os.listdir(cluster_dir)[:3]:
            img = Image.open(os.path.join(cluster_dir,img_name)).resize((120,120))
            canvas.paste(img, (x,150))
            x += 130
        canvas.save(os.path.join(out, f"cluster_{int(cid):02d}.png"))


def save_detailed_results(results, output_path="results.json", lambda_consensus=0.2, tag_tuner=0.5, losses=None, accuracy=None, epochs=None):
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
    output = {'epochs': epochs if epochs is not None else "Not provided",
              'training_losses': losses if losses is not None else "Not provided",
              'final_accuracy': accuracy if accuracy is not None else "Not provided", 'results': results,
              'hyperparams': {
                  "lambda_consensus": lambda_consensus,
                  "tag_tuner": tag_tuner
              }}
    summary = {
        "final_acc": results["metrics"]["acc"],
        "final_nmi": results["metrics"]["nmi"],
        "final_silhouette": results["metrics"]["silhouette"]
    }
    output.update(summary)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4, default=default_serializer)

    logging.info(f"Results saved to {output_path}")

def plot_experiment_results(output_dir, mode, losses, embeddings, clusters):
    """
    Saves training loss curve and PCA 2D scatter of embeddings.

    Args:
        output_dir (str): Directory to save the plots.
        mode (str): Experiment mode name ("baseline", "cae", "concat", etc.).
        losses (list or np.ndarray): Training loss per epoch.
        embeddings (np.ndarray): Embedding matrix (N x D).
        clusters (np.ndarray): Cluster assignments for each sample.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot loss curve
    if losses is not None and len(losses) > 0:
        plt.figure(dpi=120)
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title(f"Training Loss per Epoch ({mode})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        loss_path = os.path.join(output_dir, f"results_{mode}_loss.png")
        plt.savefig(loss_path, bbox_inches='tight')
        plt.close()
        logging.info(f"[Plot] Saved loss curve - {loss_path}")

    # PCA 2D scatter of embeddings
    if embeddings is not None and clusters is not None:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        plt.figure(dpi=120)
        plt.scatter(reduced[:, 0], reduced[:, 1], c=clusters, s=20)
        plt.title(f"PCA of Clustering Embeddings ({mode})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        scatter_path = os.path.join(output_dir, f"results_{mode}_pca.png")
        plt.savefig(scatter_path, bbox_inches='tight')
        plt.close()
        logging.info(f"[Plot] Saved PCA scatter - {scatter_path}")

def generate_notebook(results_file, output_notebook):
    """
    Generates a Jupyter notebook to present the results and the steps taken to achieve them.
    
    Args:
    - results_file (str): Path to the JSON file containing the results (e.g., clustering results, embeddings, etc.).
    - output_notebook (str): Path to save the generated notebook.
    """
    logging.info(f"Generating notebook at {output_notebook}")
    
    nb = nbf.v4.new_notebook()

    #Add introduction markdown cell
    intro_text = """# Results Notebook

This notebook presents the results of the clustering process performed on the AwA2 dataset using autoencoders and KMeans clustering.

## Table of Contents:
1. Data Loading
2. Model Training
3. Clustering and Embeddings
4. Visualization of Results
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(intro_text))

    #Add a code cell for loading data
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

    #Add a code cell for plotting loss curves
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

    #Add a code cell for clustering results visualization
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

    #Add embedding visualization (e.g., PCA or t-SNE)
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

    #Add a summary markdown cell
    summary_text = """
## Summary

- We trained an autoencoder on the AwA2 dataset and extracted embeddings.
- The embeddings were clustered using KMeans with the number of clusters set to 50.
- Loss per epoch was tracked, and the cluster distribution and embeddings were visualized.
    """
    nb['cells'].append(nbf.v4.new_markdown_cell(summary_text))

    #Save the notebook
    with open(output_notebook, 'w') as f:
        nbf.write(nb, f)

    logging.info(f"Notebook saved to {output_notebook}")

