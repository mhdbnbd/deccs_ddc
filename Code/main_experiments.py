"""
DDECCS: Deep Descriptive Clustering with Consensus Representations

Pipeline:
  1. Extract frozen ResNet-101 features (2048-dim)
  2. Cluster: K-means baseline or DECCS consensus ensemble
  3. Explain: ILP-based concise, orthogonal cluster descriptions (DDC)
  4. Evaluate: NMI, ACC, ARI, Silhouette

Modes:
  kmeans  — K-means baseline
  deccs   — DECCS consensus clustering
  ddc     — K-means + ILP descriptions
  ddeccs  — DECCS consensus + ILP descriptions (thesis contribution)

Datasets:
  awa2 — Animals with Attributes 2 (50 classes, 85 attributes)
  apy  — Attribute Pascal & Yahoo (32 classes, 64 attributes)
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
import pulp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import AttributeDataset, get_dataset_paths, DATASET_CONFIGS
from utils import (
    custom_collate, create_sample_dataset, setup_logging,
    get_base_clusterings, build_sparse_consensus, clustering_acc,
    summarize_clusters_with_attributes, generate_cluster_report,
)
from visualize import plot_tsne, plot_pca, save_cluster_examples


# =========================================================================
# Feature extraction
# =========================================================================

def extract_resnet_features(dataloader, device):
    """Extract frozen ResNet-101 features (2048-dim) for the entire dataset."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    backbone.eval()

    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting ResNet-101 features"):
            if batch is None:
                continue
            features = backbone(batch[0].to(device)).flatten(1)
            all_features.append(features.cpu())

    del backbone, resnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.cat(all_features).numpy()


# =========================================================================
# ILP cluster descriptions (DDC)
# =========================================================================

def solve_ilp_descriptions(cluster_labels, tags, n_clusters, alpha=8, predicate_names=None):
    """
    DDC's ILP (Eq. 2-4): concise, orthogonal cluster descriptions.

    Returns list of dicts: [{cluster_id, n_members, tags, tag_indices}, ...]
    """

    M = tags.shape[1]
    Q = np.zeros((n_clusters, M))
    for k in range(n_clusters):
        members = (cluster_labels == k)
        if members.sum() > 0:
            Q[k] = tags[members].mean(axis=0)

    active = [k for k in range(n_clusters) if (cluster_labels == k).sum() > 0]
    K_act = len(active)
    Q_act = Q[active]

    logging.info(f"[ILP] {K_act} active clusters, {M} attributes, alpha={alpha}")

    # Search for smallest feasible beta
    W_np = None
    for beta in range(1, K_act + 1):
        prob = pulp.LpProblem("DDC_ILP", pulp.LpMinimize)
        W = {(i, j): pulp.LpVariable(f"W_{i}_{j}", cat='Binary')
             for i in range(K_act) for j in range(M)}

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

            n_tags = int((W_np.sum(axis=0) > 0.5).sum())
            logging.info(f"[ILP] Solved: beta={beta}, {n_tags}/{M} tags selected, "
                         f"{W_np.sum(1).mean():.1f} tags/cluster")
            break

    if W_np is None:
        logging.warning("[ILP] Infeasible — greedy fallback")
        W_np = np.zeros((K_act, M))
        usage = np.zeros(M)
        for i in range(K_act):
            scores = Q_act[i] / (1.0 + usage)
            top = np.argsort(scores)[-alpha:]
            W_np[i, top] = 1
            usage[top] += 1

    # Build readable descriptions with TC and ITF metrics (DDC Eq. 8-9)
    pred_names = predicate_names or [f"attr_{j}" for j in range(M)]
    K_total = K_act  # for ITF denominator
    descriptions = []

    # Precompute: which tags are used by which clusters (for ITF)
    tag_cluster_count = np.zeros(M)
    for idx in range(K_act):
        for j in range(M):
            if W_np[idx, j] > 0.5:
                tag_cluster_count[j] += 1

    for idx, k in enumerate(active):
        selected = np.where(W_np[idx] > 0.5)[0]
        tag_names = [pred_names[j] for j in selected]
        n_members = int((cluster_labels == k).sum())
        members_mask = (cluster_labels == k)

        # Tag Coverage (DDC Eq. 8): fraction of cluster members having each tag
        if len(selected) > 0 and members_mask.sum() > 0:
            coverage_per_tag = []
            for j in selected:
                frac = tags[members_mask, j].mean()
                coverage_per_tag.append(float(frac))
            tc = float(np.mean(coverage_per_tag))
        else:
            tc = 0.0

        # Inverse Tag Frequency (DDC Eq. 9): higher = more unique tags
        if len(selected) > 0 and K_total > 0:
            itf_per_tag = []
            for j in selected:
                n_clusters_using_j = max(tag_cluster_count[j], 1)
                itf_per_tag.append(float(np.log(K_total / n_clusters_using_j)))
            itf = float(np.mean(itf_per_tag))
        else:
            itf = 0.0

        descriptions.append({
            "cluster_id": int(k),
            "n_members": n_members,
            "tags": tag_names,
            "tag_indices": selected.tolist(),
            "tc": round(tc, 4),
            "itf": round(itf, 4),
        })
        logging.info(f"  C{k:2d} (n={n_members:5d}): {', '.join(tag_names[:6]):50s} TC={tc:.2f} ITF={itf:.2f}")

    # Aggregate TC and ITF
    avg_tc = np.mean([d["tc"] for d in descriptions])
    avg_itf = np.mean([d["itf"] for d in descriptions])
    logging.info(f"  Average TC={avg_tc:.4f}, Average ITF={avg_itf:.4f}")

    return descriptions, W_np, {"avg_tc": round(avg_tc, 4), "avg_itf": round(avg_itf, 4)}


def load_predicate_names(predicates_path):
    """Load human-readable attribute names from predicates.txt."""
    names = []
    if os.path.exists(predicates_path):
        with open(predicates_path) as f:
            for line in f:
                parts = line.strip().split('\t') if '\t' in line else line.strip().split()
                names.append(parts[-1] if parts else "?")
    return names


# =========================================================================
# Main pipeline
# =========================================================================

def select_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def main():
    parser = argparse.ArgumentParser(
        description="DDECCS: Deep Descriptive Clustering with Consensus Representations")
    parser.add_argument("--dataset", type=str, choices=["awa2", "apy"],
                        default="awa2")
    parser.add_argument("--mode", type=str,
                        choices=["kmeans", "deccs", "ddc", "ddeccs"],
                        required=True)
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters (default: number of classes)")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--sample_size", type=int, default=2000)
    args = parser.parse_args()

    setup_logging()
    device = select_device(args.use_gpu)

    # --- Output directory: results/<dataset>/<mode>/ ---
    output_dir = os.path.join("results", args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset paths ---
    paths = get_dataset_paths(args.dataset, use_sample=args.use_sample)
    cfg = DATASET_CONFIGS[args.dataset]

    if args.use_sample:
        create_sample_dataset(
            cfg["source_dir"],
            os.path.join("samples", args.dataset),
            os.path.join(cfg["source_dir"], "classes.txt"),
            sample_size=args.sample_size,
        )

    # Check if labels file exists; if not, try AwA2-labels.txt fallback
    if not os.path.exists(paths["attr_file"]):
        alt = paths["attr_file"].replace("labels.txt", "AwA2-labels.txt")
        if os.path.exists(alt):
            paths["attr_file"] = alt
        else:
            # Generate labels from directory structure
            logging.warning(f"Labels file not found: {paths['attr_file']}")
            raise FileNotFoundError(f"Labels file not found: {paths['attr_file']}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = AttributeDataset(
        img_dir=paths["img_dir"],
        attr_file=paths["attr_file"],
        pred_file=paths["pred_file"],
        classes_file=paths["classes_file"],
        transform=transform, train=True,
    )

    K = args.n_clusters or len(np.unique(dataset.labels))
    logging.info(f"=== {cfg['name']} | Mode: {args.mode.upper()} | K={K} | N={len(dataset)} ===")

    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4,
        pin_memory=True, collate_fn=custom_collate, shuffle=False,
    )

    # =====================================================================
    # Step 1: Extract features
    # =====================================================================
    start = time.time()
    features = extract_resnet_features(dataloader, device)
    true_labels = np.array(dataset.labels)
    symbolic_tags = dataset.symbolic_tags
    feat_time = time.time() - start
    logging.info(f"Features: {features.shape} ({feat_time:.1f}s)")

    # =====================================================================
    # Step 2: Clustering
    # =====================================================================
    t_cluster = time.time()
    if args.mode in ["kmeans", "ddc"]:
        logging.info(f"K-means (K={K})...")
        cluster_labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(features)
    else:
        logging.info(f"DECCS consensus (K={K})...")
        base_labels = get_base_clusterings(features, n_clusters=K)
        consensus = build_sparse_consensus(base_labels, features)
        cluster_labels = SpectralClustering(
            n_clusters=K, affinity="precomputed",
            assign_labels="kmeans", random_state=42,
        ).fit_predict(consensus)

    n_actual = len(np.unique(cluster_labels))
    logging.info(f"Clustering: {n_actual} clusters ({time.time()-t_cluster:.1f}s)")

    # =====================================================================
    # Step 3: ILP Descriptions
    # =====================================================================
    ilp_descriptions = None
    ilp_metrics = {}
    if args.mode in ["ddc", "ddeccs"]:
        pred_names = load_predicate_names(paths["predicates_file"])
        ilp_descriptions, W_expl, ilp_metrics = solve_ilp_descriptions(
            cluster_labels, symbolic_tags, n_clusters=K, alpha=8,
            predicate_names=pred_names,
        )

        with open(os.path.join(output_dir, "ilp_descriptions.json"), "w") as f:
            json.dump(ilp_descriptions, f, indent=2)

    # =====================================================================
    # Step 4: Evaluation
    # =====================================================================
    acc = clustering_acc(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(features, cluster_labels,
                           sample_size=min(10000, len(features)), random_state=42)

    metrics = {"nmi": nmi, "acc": acc, "ari": ari, "silhouette": sil, **ilp_metrics}
    logging.info(f"NMI={nmi:.4f}  ACC={acc:.4f}  ARI={ari:.4f}  Sil={sil:.4f}"
                 + (f"  TC={ilp_metrics.get('avg_tc',0):.4f}  ITF={ilp_metrics.get('avg_itf',0):.4f}"
                    if ilp_metrics else ""))

    # =====================================================================
    # Step 5: Save everything
    # =====================================================================

    # Attribute-based cluster summaries
    cluster_descriptions = []
    for c in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == c
        cluster_descriptions.append(
            np.mean(symbolic_tags[mask], axis=0) if mask.sum() > 0
            else np.zeros(symbolic_tags.shape[1])
        )
    cluster_descriptions = np.array(cluster_descriptions)

    summarize_clusters_with_attributes(
        cluster_labels=cluster_labels, cluster_descriptions=cluster_descriptions,
        dataset=dataset, predicates_path=paths["predicates_file"],
        top_k=5, output_json=os.path.join(output_dir, "cluster_descriptions.json"),
    )

    # Visualizations
    plot_tsne(features, cluster_labels,
              os.path.join(output_dir, "tsne.png"),
              title=f"{cfg['name']} — {args.mode.upper()} (NMI={nmi:.3f})")
    plot_pca(features, cluster_labels,
             os.path.join(output_dir, "pca.png"),
             title=f"{cfg['name']} — {args.mode.upper()} PCA (NMI={nmi:.3f})")

    # Cluster sample images
    save_cluster_examples(
        cluster_labels, dataset,
        output_dir=os.path.join(output_dir, "cluster_samples"),
    )

    # Summary JSON
    summary = {
        "dataset": args.dataset,
        "dataset_name": cfg["name"],
        "mode": args.mode,
        "n_clusters": K,
        "n_samples": len(dataset),
        "n_classes": len(np.unique(true_labels)),
        "n_attributes": symbolic_tags.shape[1],
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "cluster_sizes": {
            "min": int(np.bincount(cluster_labels).min()),
            "max": int(np.bincount(cluster_labels).max()),
            "mean": round(np.bincount(cluster_labels).mean(), 1),
        },
        "device": str(device),
        "time_seconds": round(time.time() - start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Full results (with cluster assignments)
    full_results = {**summary, "clusters": cluster_labels.tolist()}
    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    logging.info(f"Results saved to {output_dir}/")
    logging.info(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
