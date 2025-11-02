import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.tight_layout()
    plt.scatter(reduced[:,0], reduced[:,1], c=labels, s=8)
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

