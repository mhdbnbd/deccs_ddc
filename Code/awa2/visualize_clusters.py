import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

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
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y, s=8, cmap="tab20")
    plt.title("t-SNE projection of DECCS embeddings", fontsize=14)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] t-SNE plot saved to {save_path}")

def save_cluster_examples(cluster_assignments, dataset, n_per_cluster=3, outdir="cluster_samples"):
    import os, shutil, logging
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for idx, c in enumerate(cluster_assignments):
        if counts.get(c, 0) < n_per_cluster:
            img_path = dataset.image_paths[idx]
            lbl = dataset.labels[idx]
            cls_dir = os.path.join(outdir, f"cluster_{c:02d}")
            os.makedirs(cls_dir, exist_ok=True)
            if not os.access(cls_dir, os.W_OK):
                logging.warning(f"Cannot write to {cls_dir}, skipping.")
                continue
            shutil.copy(img_path, os.path.join(cls_dir, os.path.basename(img_path)))
            counts[c] = counts.get(c, 0) + 1
            logging.info(f"Cluster {c:02d}: saved {os.path.basename(img_path)} (label={lbl})")

