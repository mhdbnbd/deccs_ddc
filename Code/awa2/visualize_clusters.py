import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
import os
import shutil
import stat
import datetime

def plot_tsne(embeddings, labels, save_path, max_points=8000):
    """
    t-SNE visualization for embeddings.
    with automatic subsampling.
    """
    X = embeddings
    y = np.array(labels)

    # --- Subsample if needed ---
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X[idx]
        y = y[idx]

    # --- Reduce dimensionality before t-SNE ---
    X50 = PCA(n_components=50, random_state=42).fit_transform(X)

    # --- Run t-SNE on reduced embeddings ---
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    reduced = tsne.fit_transform(X50)

    # --- Plot the subsampled data ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=y, s=8, cmap="tab20", alpha=0.8)
    plt.title("t-SNE projection of DECCS embeddings", fontsize=14)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] t-SNE plot saved to {save_path}")

def ensure_dir_writable(path):
    """
    Ensure the directory exists and has full read/write/execute permissions
    """
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except PermissionError:
        logging.warning(f"Could not modify permissions for {path}. Check user rights.")

def get_safe_output_dir(base_dir="cluster_samples", mode=None):
    """
    Create a unique, mode-specific output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode:
        outdir = os.path.join(base_dir, f"{mode}_{timestamp}")
    else:
        outdir = os.path.join(base_dir, timestamp)
    ensure_dir_writable(outdir)
    return outdir

def save_cluster_examples(cluster_assignments, dataset, n_per_cluster=3, outdir="cluster_samples", mode=None):
    """
    Save representative samples from each cluster into a structured, mode-safe folder.

    Args:
        cluster_assignments (list or np.ndarray): Cluster labels for each sample.
        dataset: Dataset object containing `image_paths` and `labels`.
        n_per_cluster (int): Number of images to save per cluster.
        outdir (str): Base output directory for saving cluster samples.
        mode (str): Optional experiment mode (e.g., 'ae', 'cae', 'deccs').
    """
    # --- Create a safe subdirectory to avoid conflicts ---
    save_root = get_safe_output_dir(outdir, mode)
    counts = {}

    for idx, cluster_id in enumerate(cluster_assignments):
        # Skip invalid cluster labels (e.g., -1 from DBSCAN)
        if cluster_id < 0:
            continue

        cls_dir = os.path.join(save_root, f"cluster_{int(cluster_id):02d}")
        ensure_dir_writable(cls_dir)

        # Limit saved examples per cluster
        if counts.get(cluster_id, 0) >= n_per_cluster:
            continue

        try:
            img_path = dataset.image_paths[idx]
            label = dataset.labels[idx]
            dest_path = os.path.join(cls_dir, os.path.basename(img_path))

            # Overwrite-safe copy
            if os.path.exists(dest_path):
                os.remove(dest_path)

            shutil.copy(img_path, dest_path)
            counts[cluster_id] = counts.get(cluster_id, 0) + 1
            logging.info(f"Cluster {cluster_id:02d}: saved {os.path.basename(img_path)} (label={label})")

        except PermissionError:
            logging.warning(f"Permission denied when saving to {cls_dir}. Skipping {img_path}.")
        except Exception as e:
            logging.warning(f"Failed to save image {img_path}: {e}")

    logging.info(f"Cluster sample saving complete — results stored in: {save_root}")

def plot_deccs_loss(log_path="results_deccs_loss_components.npz", save_path="results_deccs_loss.png"):

    data = np.load(log_path)
    plt.figure()
    epochs = np.arange(1, len(data["total"]) + 1)
    for e in range(5, len(epochs), 5):
        plt.axvline(x=e, color="gray", linestyle="--", alpha=0.3)
    plt.plot(epochs, data["total"], label="Total Loss")
    plt.plot(epochs, data["recon"], label="Reconstruction Loss", alpha=0.7)
    plt.plot(epochs, data["tag"], label="Tag Loss", alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("DECCS Training Loss (component-wise)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
