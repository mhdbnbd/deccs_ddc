import json
import logging
import os
import random
import shutil
import time
from logging.handlers import RotatingFileHandler

import numpy as np
from PIL import Image, ImageDraw
# from clustpy.deep import DEC
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_matrix
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, \
    silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
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

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([item[0] for item in batch])
    tags = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    indices = torch.tensor([item[3] for item in batch])
    return images, tags, labels, indices

def build_sparse_consensus(base_labels, embeddings_np, k=20):
    n_clusterers, n_samples = base_labels.shape
    A = lil_matrix((n_samples, n_samples), dtype=np.float32)

    # Build kNN graph once (on features used to produce base_labels ideally, here we proxy with labels)
    # We approximate: connect each point to its k nearest by index; for real kNN use the feature matrix.
    X_norm = StandardScaler().fit_transform(embeddings_np)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nbrs.fit(X_norm)
    _, knn_idx = nbrs.kneighbors(X_norm)

    for labels in base_labels:
        for i in range(n_samples):
            same = (labels[knn_idx[i]] == labels[i]).astype(np.float32)
            A[i, knn_idx[i]] += same
    A = (A + A.T).tocsr()
    A /= float(n_clusterers)
    A = sort_graph_by_row_values(A, warn_when_not_sorted=False)
    return A



def get_base_clusterings(embeddings_np, n_clusters=10, seed=42):
    """
    Build an ensemble of classical clusterers for DECCS consensus.

    Classical: KMeans, Spectral, GMM, Agglomerative  (from scikit-learn)

    DBSCAN is excluded: in high-dimensional embedding spaces (d=128),
    fixed-eps DBSCAN labels all points as noise, which corrupts the
    consensus matrix by falsely co-associating all noise pairs.
    Neither DECCS nor DDC use DBSCAN in their ensembles.
    """

    # --- Normalize embeddings ---
    X = StandardScaler().fit_transform(embeddings_np)

    # --- PCA for GMM: full covariance on 2048-dim is infeasible ---
    n_gmm_dim = min(256, X.shape[1], max(64, X.shape[0] // (n_clusters * 2)))
    if X.shape[1] > n_gmm_dim:
        from sklearn.decomposition import PCA as PCA_sk
        X_gmm = PCA_sk(n_components=n_gmm_dim,  random_state=seed).fit_transform(X)
        logging.info(f"[DECCS] GMM uses PCA-reduced features: {X.shape[1]} → {n_gmm_dim}")
    else:
        X_gmm = X

    # --- Prepare ensemble containers ---
    base_labels = []
    classical_models = {
        "KMeans": (KMeans(
            n_clusters=n_clusters,
            n_init='auto',
            max_iter=300,
            algorithm="elkan",
             random_state=seed
        ), X),
        "Spectral": (SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=15,
            assign_labels="kmeans",
            eigen_solver="arpack",
             random_state=seed
        ), X),
        "GMM": (GaussianMixture(n_components=n_clusters,  random_state=seed,
                                covariance_type="full", reg_covar=1e-3, max_iter=300), X_gmm),
        "Agglomerative": (AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            compute_full_tree=False
        ), X),
    }

    # --- Run classical clustering ensemble ---
    for name, (algo, data) in classical_models.items():
        start = time.time()
        try:
            labels = algo.fit_predict(data)
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

def clustering_acc(y_true, y_pred):
    y_true_u, y_true = np.unique(y_true, return_inverse=True)
    y_pred_u, y_pred = np.unique(y_pred, return_inverse=True)
    D = np.zeros((y_true_u.size, y_pred_u.size), dtype=np.int64)
    for i in range(y_true.size):
        D[y_true[i], y_pred[i]] += 1
    r, c = linear_sum_assignment(D.max() - D)
    return D[r, c].sum() / y_true.size

# Convert all numpy/tensor values to standard Python types
def load_attribute_names(predicates_path="data/AwA2-data/Animals_with_Attributes2/predicates.txt"):
    """
    Load attribute names from predicates.txt.
    Handles both AwA2 (single-word names) and aPY (multi-word names like 'Jet engine').
    """
    if not os.path.exists(predicates_path):
        raise FileNotFoundError(f"Predicates file not found at {predicates_path}")

    attribute_names = []
    with open(predicates_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "index\tname" or "index name"
            if '\t' in line:
                parts = line.split('\t', 1)
                attribute_names.append(parts[1] if len(parts) > 1 else parts[0])
            else:
                parts = line.split(None, 1)
                attribute_names.append(parts[1] if len(parts) > 1 else parts[0])
    return attribute_names

def default_serializer(o):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    elif isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    elif hasattr(o, 'tolist'):
        return o.tolist()
    else:
        return str(o)


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
        readable_attrs = [attribute_names[j] if j < len(attribute_names) else f"attr_{j}" for j in top_attrs]

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
    return results
