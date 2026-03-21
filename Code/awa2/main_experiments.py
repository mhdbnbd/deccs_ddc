"""
DECCS-DDC Pipeline: Consensus Clustering with Interpretable Descriptions

Pipeline:
  1. Extract frozen ResNet-101 features (2048-dim)
  2. Cluster using K-means or DECCS consensus ensemble
  3. Generate interpretable cluster descriptions via ILP (DDC's contribution)
  4. Evaluate with NMI, ACC, ARI, Silhouette

Modes:
  - kmeans:  K-means on ResNet features (baseline)
  - deccs:   DECCS consensus clustering on ResNet features
  - ddc:     K-means + ILP cluster descriptions (DDC interpretability)
  - ddeccs:  DECCS consensus + ILP descriptions (thesis contribution)
"""

import argparse
import json
import logging
import os
import random
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import AwA2Dataset
from utils import (
    custom_collate, create_sample_dataset, setup_logging,
    save_detailed_results, get_base_clusterings, build_sparse_consensus,
    clustering_acc, summarize_clusters_with_attributes, generate_cluster_report
)
from visualize_clusters import plot_tsne, save_cluster_examples


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def extract_resnet_features(dataloader, device):
    """Extract frozen ResNet-101 features for the entire dataset."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    backbone.eval()

    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting ResNet-101 features"):
            if batch is None:
                continue
            images = batch[0].to(device)
            features = backbone(images).flatten(1)
            all_features.append(features.cpu())

    del backbone, resnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(all_features).numpy()


def solve_ilp_descriptions(cluster_labels, tags, n_clusters, alpha=8):
    """
    DDC's ILP (Eq. 2-4): find concise, orthogonal cluster descriptions.

    Uses PuLP CBC solver. For each cluster, selects a small set of tags
    that covers the cluster well while being orthogonal to other clusters.

    Returns:
        W: (K_active, M) binary explanation matrix
        mask: (M,) binary — which tags are used by any cluster
        active: list of active cluster indices
    """
    import pulp

    M = tags.shape[1]

    Q = np.zeros((n_clusters, M))
    for k in range(n_clusters):
        members = (cluster_labels == k)
        if members.sum() > 0:
            Q[k] = tags[members].mean(axis=0)

    active = [k for k in range(n_clusters) if (cluster_labels == k).sum() > 0]
    K_act = len(active)
    Q_act = Q[active]

    logging.info(f"[ILP] Solving for {K_act} active clusters, {M} tags, alpha={alpha}")

    for beta in range(1, K_act + 1):
        prob = pulp.LpProblem("DDC_ILP", pulp.LpMinimize)

        W = {}
        for i in range(K_act):
            for j in range(M):
                W[i, j] = pulp.LpVariable(f"W_{i}_{j}", cat='Binary')

        prob += pulp.lpSum(W[i, j] for i in range(K_act) for j in range(M))

        for i in range(K_act):
            prob += pulp.lpSum(W[i, j] * Q_act[i, j] for j in range(M)) >= alpha

        for j in range(M):
            prob += pulp.lpSum(W[i, j] * Q_act[i, j] for i in range(K_act)) <= beta

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

        if prob.status == 1:
            W_np = np.zeros((K_act, M))
            for i in range(K_act):
                for j in range(M):
                    W_np[i, j] = W[i, j].varValue or 0

            mask = (W_np.sum(axis=0) > 0.5).astype(np.float32)
            n_tags = int(mask.sum())
            logging.info(f"[ILP] Solved: beta={beta}, {n_tags}/{M} tags, "
                         f"{W_np.sum(1).mean():.1f} tags/cluster")
            return W_np, mask, active

    logging.warning("[ILP] No feasible solution — greedy fallback")
    W_np = np.zeros((K_act, M))
    usage = np.zeros(M)
    for i in range(K_act):
        scores = Q_act[i] / (1.0 + usage)
        top = np.argsort(scores)[-alpha:]
        W_np[i, top] = 1
        usage[top] += 1
    mask = (W_np.sum(axis=0) > 0.5).astype(np.float32)
    return W_np, mask, active


setup_logging()
device = select_device()


def main():
    parser = argparse.ArgumentParser(description="DECCS-DDC Clustering Pipeline")
    parser.add_argument("--mode", type=str,
                        choices=["kmeans", "deccs", "ddc", "ddeccs"],
                        required=True,
                        help="kmeans: baseline, deccs: consensus, "
                             "ddc: kmeans+ILP, ddeccs: consensus+ILP (thesis)")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    run_tag = f"{args.mode}_resnet"
    if args.output_json is None:
        args.output_json = f"results_{run_tag}.json"

    logging.info(f"=== Mode: {args.mode.upper()} | K={args.n_clusters} ===")

    # --- Dataset ---
    source_dir = "data/AwA2-data/Animals_with_Attributes2"
    dataset_dir = "AwA2-sample"
    pred_file = os.path.join(source_dir, "predicate-matrix-continuous.txt")
    classes_file = os.path.join(source_dir, "classes.txt")

    if args.use_sample:
        create_sample_dataset(source_dir, dataset_dir, classes_file, sample_size=args.sample_size)
        img_dir = os.path.join(dataset_dir, "JPEGImages")
        attr_file = os.path.join(dataset_dir, "AwA2-labels.txt")
    else:
        img_dir = os.path.join(source_dir, "JPEGImages")
        attr_file = os.path.join(source_dir, "AwA2-labels.txt")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = AwA2Dataset(
        img_dir=img_dir, attr_file=attr_file, pred_file=pred_file,
        classes_file=classes_file, transform=transform, train=True
    )

    n_classes = len(np.unique(dataset.labels))
    K = args.n_clusters
    logging.info(f"Dataset: {len(dataset)} samples, {n_classes} classes, K={K}")

    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4,
        pin_memory=True, collate_fn=custom_collate, shuffle=False
    )

    # =========================================================================
    # Step 1: Extract frozen ResNet-101 features
    # =========================================================================
    start = time.time()
    features = extract_resnet_features(dataloader, device)
    true_labels = np.array(dataset.labels)
    symbolic_tags = dataset.symbolic_tags
    logging.info(f"Features: {features.shape} in {time.time()-start:.1f}s")

    # =========================================================================
    # Step 2: Clustering
    # =========================================================================
    if args.mode in ["kmeans", "ddc"]:
        logging.info(f"Running K-means (K={K})...")
        cluster_labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(features)

    elif args.mode in ["deccs", "ddeccs"]:
        logging.info(f"Running DECCS consensus (K={K})...")
        base_labels = get_base_clusterings(features, n_clusters=K)
        consensus = build_sparse_consensus(base_labels, features)
        cluster_labels = SpectralClustering(
            n_clusters=K, affinity="precomputed",
            assign_labels="kmeans", random_state=42
        ).fit_predict(consensus)

    logging.info(f"Clustering complete: {len(np.unique(cluster_labels))} clusters")

    # =========================================================================
    # Step 3: ILP Cluster Descriptions (DDC interpretability)
    # =========================================================================
    if args.mode in ["ddc", "ddeccs"]:
        logging.info("Generating ILP cluster descriptions...")
        W_expl, tag_mask, active_clusters = solve_ilp_descriptions(
            cluster_labels, symbolic_tags, n_clusters=K, alpha=8
        )

        pred_path = os.path.join(source_dir, "predicates.txt")
        pred_names = []
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                for line in f:
                    parts = line.strip().split('\t') if '\t' in line else line.strip().split()
                    pred_names.append(parts[-1] if parts else "?")

        if W_expl is not None and len(pred_names) == symbolic_tags.shape[1]:
            logging.info("=== ILP Cluster Descriptions ===")
            for idx, k in enumerate(active_clusters):
                selected = np.where(W_expl[idx] > 0.5)[0]
                tag_names = [pred_names[j] for j in selected]
                n_members = (cluster_labels == k).sum()
                logging.info(f"  Cluster {k:2d} (n={n_members:5d}): {', '.join(tag_names)}")

    # =========================================================================
    # Step 4: Evaluation
    # =========================================================================
    acc = clustering_acc(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(features, cluster_labels,
                           sample_size=min(10000, len(features)), random_state=42)

    logging.info(f"Results: ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil:.4f}")

    results = {
        "acc": float(acc), "ari": float(ari), "nmi": float(nmi),
        "silhouette": float(sil), "clusters": cluster_labels.tolist()
    }

    # =========================================================================
    # Step 5: Visualization & Saving
    # =========================================================================
    cluster_descriptions = []
    for c in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == c
        cluster_descriptions.append(np.mean(symbolic_tags[mask], axis=0) if mask.sum() > 0
                                    else np.zeros(symbolic_tags.shape[1]))
    cluster_descriptions = np.array(cluster_descriptions)

    summarize_clusters_with_attributes(
        cluster_labels=cluster_labels, cluster_descriptions=cluster_descriptions,
        dataset=dataset,
        predicates_path=os.path.join(source_dir, "predicates.txt"),
        top_k=5, output_json=f"results_{run_tag}_cluster_descriptions.json"
    )

    plot_tsne(features, cluster_labels, f"results_{run_tag}_tsne.png")
    samples_dir = save_cluster_examples(cluster_labels, dataset, mode=run_tag)

    save_detailed_results(
        results={"mode": args.mode, "arch": "resnet101_frozen", "metrics": results},
        output_path=args.output_json, lambda_consensus=0, tag_tuner=0
    )

    report = {
        "metrics": {k: v for k, v in results.items() if k != "clusters"},
        "mode": args.mode, "n_clusters": K,
        "n_samples": len(dataset), "n_classes": n_classes,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"results_{run_tag}_summary.json", "w") as f:
        json.dump(report, f, indent=4)

    generate_cluster_report(
        samples_dir=samples_dir,
        descriptions_json=f"results_{run_tag}_cluster_descriptions.json"
    )

    logging.info(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()