import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, save_path, max_points=8000):
    # downsample to PCA-50
    X = embeddings
    y = labels
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X[idx];
        y = np.array(y)[idx]

    X50 = PCA(n_components=50, random_state=42).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    reduced = tsne.fit_transform(X50)
    plt.tight_layout()
    plt.scatter(reduced[:,0], reduced[:,1], c=labels, s=84)
    plt.title("t-SNE projection of DECCS embeddings")
    plt.savefig(save_path, dpi=300)
    plt.close()

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
            shutil.copy(img_path, cls_dir)
            counts[c] = counts.get(c, 0) + 1
            logging.info(f"Cluster {c:02d}: saved {os.path.basename(img_path)} (label={lbl})")

