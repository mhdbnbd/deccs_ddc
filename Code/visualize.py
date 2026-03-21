"""Visualization utilities for clustering results."""

import logging
import os
import shutil
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, save_path, max_points=8000, title=None):

    X, y = embeddings, np.array(labels)

    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X, y = X[idx], y[idx]

    n_components = min(50, X.shape[1], X.shape[0])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(X_pca)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=8, cmap="tab20", alpha=0.8)
    plt.title(title or "t-SNE Cluster Visualization", fontsize=14)
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logging.info(f"t-SNE plot saved to {save_path}")


def plot_pca(embeddings, labels, save_path, title=None):

    X, y = embeddings, np.array(labels)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=8, cmap="tab20", alpha=0.8)
    plt.title(title or "PCA Cluster Visualization", fontsize=14)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logging.info(f"PCA plot saved to {save_path}")


def save_cluster_examples(cluster_labels, dataset, output_dir, n_per_cluster=3):
    """Save representative images from each cluster."""
    os.makedirs(output_dir, exist_ok=True)
    counts = {}

    for idx, cid in enumerate(cluster_labels):
        if cid < 0 or counts.get(cid, 0) >= n_per_cluster:
            continue

        cls_dir = os.path.join(output_dir, f"cluster_{int(cid):02d}")
        os.makedirs(cls_dir, exist_ok=True)

        try:
            src = dataset.image_paths[idx]
            dst = os.path.join(cls_dir, os.path.basename(src))
            shutil.copy(src, dst)
            counts[cid] = counts.get(cid, 0) + 1
        except Exception as e:
            logging.warning(f"Failed to save {src}: {e}")

    logging.info(f"Cluster examples saved to {output_dir}")
    return output_dir