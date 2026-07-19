"""
DDECCS: Deep Descriptive Clustering with Consensus Representations

Pipeline:
  1. Extract frozen ResNet-101 features (2048-dim), cached to disk
  2. Optional PCA dimensionality reduction
  3. Cluster: K-means baseline or DECCS consensus ensemble
  4. Explain: ILP-based concise, orthogonal cluster descriptions (DDC)
  5. Evaluate: NMI, ACC, ARI, Silhouette, TC, ITF
  6. Multi-seed runs for mean±std reporting (matching DDC paper)

Modes:
  kmeans  — K-means baseline
  deccs   — DECCS consensus clustering
  ddc     — K-means + ILP descriptions
  ddeccs  — DECCS consensus + ILP descriptions (thesis contribution)
"""

import argparse
import json
import logging
import os
import random
import time
import hashlib

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from dataset import AttributeDataset, get_dataset_paths, DATASET_CONFIGS, APY_DDC_15_CLASSES
from utils import (
    custom_collate, create_sample_dataset, setup_logging,
    get_base_clusterings, build_sparse_consensus, clustering_acc,
    summarize_clusters_with_attributes, generate_cluster_report,
)
from visualize import plot_tsne, plot_pca, save_cluster_examples


# =========================================================================
# Feature extraction with disk caching
# =========================================================================

def get_cache_path(dataset_name, use_sample, sample_size):
    """Deterministic cache path for extracted features."""
    tag = f"{dataset_name}_{'sample' + str(sample_size) if use_sample else 'full'}"
    return os.path.join("cache", f"resnet101_{tag}.npz")


def extract_resnet_features(dataloader, device, cache_path=None):
    """Extract frozen ResNet-101 features, with optional disk caching."""

    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        logging.info(f"Loaded cached features from {cache_path}")
        return data["features"]

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

    features = torch.cat(all_features).numpy()

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, features=features)
        logging.info(f"Cached features to {cache_path}")

    return features


# =========================================================================
# ILP cluster descriptions (DDC)
# =========================================================================

def solve_ilp_descriptions(cluster_labels, tags, n_clusters, alpha=8,
                           predicate_names=None):
    """
    DDC's ILP (Eq. 2-4): concise, orthogonal cluster descriptions.
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

    # TC and ITF (DDC Eq. 8-9)
    pred_names = predicate_names or [f"attr_{j}" for j in range(M)]
    K_total = K_act
    tag_cluster_count = np.zeros(M)
    for idx in range(K_act):
        for j in range(M):
            if W_np[idx, j] > 0.5:
                tag_cluster_count[j] += 1

    descriptions = []
    for idx, k in enumerate(active):
        selected = np.where(W_np[idx] > 0.5)[0]
        tag_names = [pred_names[j] for j in selected]
        n_members = int((cluster_labels == k).sum())
        members_mask = (cluster_labels == k)

        if len(selected) > 0 and members_mask.sum() > 0:
            tc = float(np.mean([tags[members_mask, j].mean() for j in selected]))
        else:
            tc = 0.0

        if len(selected) > 0 and K_total > 0:
            itf = float(np.mean([np.log(K_total / max(tag_cluster_count[j], 1))
                                 for j in selected]))
        else:
            itf = 0.0

        descriptions.append({
            "cluster_id": int(k), "n_members": n_members,
            "tags": tag_names, "tag_indices": selected.tolist(),
            "tc": round(tc, 4), "itf": round(itf, 4),
        })
        logging.info(f"  C{k:2d} (n={n_members:5d}): {', '.join(tag_names[:6]):50s} "
                     f"TC={tc:.2f} ITF={itf:.2f}")

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
                line = line.strip()
                if not line:
                    continue
                if '\t' in line:
                    parts = line.split('\t', 1)
                    names.append(parts[1] if len(parts) > 1 else parts[0])
                else:
                    parts = line.split(None, 1)
                    names.append(parts[1] if len(parts) > 1 else parts[0])
    return names


# =========================================================================
# Single run
# =========================================================================

def run_single(features, true_labels, symbolic_tags, cluster_labels_override,
               mode, K, seed, cfg, paths, output_dir, pca_dim=None,
               standardize=False):
    """Run clustering + evaluation for a single seed. Returns metrics dict."""

    feats = features.copy()

    # Optional PCA
    if pca_dim and pca_dim < feats.shape[1]:
        feats = PCA(n_components=pca_dim, random_state=seed).fit_transform(feats)
        logging.info(f"PCA: {features.shape[1]} → {pca_dim}")

    # Optional standardization (ablation). Applied before clustering so that
    # clustering and silhouette share one space. Near-no-op for deccs/ddeccs,
    # which already standardize inside get_base_clusterings.
    if standardize:
        from sklearn.preprocessing import StandardScaler
        feats = StandardScaler().fit_transform(feats)
        logging.info("Features z-scored before clustering")

    # Clustering
    if mode in ["kmeans", "ddc"]:
        cluster_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(feats)
    else:
        base_labels = get_base_clusterings(feats, n_clusters=K, seed=seed)
        consensus = build_sparse_consensus(base_labels, feats)
        cluster_labels = SpectralClustering(
            n_clusters=K, affinity="precomputed",
            assign_labels="kmeans", random_state=seed,
        ).fit_predict(consensus)

    # ILP descriptions (only for seed 0 / primary run)
    ilp_metrics = {}
    if mode in ["ddc", "ddeccs"]:
        pred_names = load_predicate_names(paths["predicates_file"])

        # Auto alpha: scale with tag density so ILP is always feasible
        # DDC uses alpha=8 on AwA2 (density ~0.5, continuous attributes).
        # For aPY (density ~0.09, binary attributes), alpha must be lower.
        # Target: ~4-6 descriptive tags per cluster.
        # Formula: alpha = clip(round(16 * density), 2, 8)
        mean_tag_density = symbolic_tags.mean()
        alpha = max(2, min(8, round(16 * mean_tag_density)))
        logging.info(f"[ILP] Tag density={mean_tag_density:.3f} → alpha={alpha}")

        ilp_descriptions, W_expl, ilp_metrics = solve_ilp_descriptions(
            cluster_labels, symbolic_tags, n_clusters=K, alpha=alpha,
            predicate_names=pred_names,
        )

        if seed == 42:  # Save descriptions only for primary run
            with open(os.path.join(output_dir, "ilp_descriptions.json"), "w") as f:
                json.dump(ilp_descriptions, f, indent=2)

    # Evaluation
    acc = clustering_acc(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(feats, cluster_labels,
                           sample_size=min(10000, len(feats)), random_state=seed)

    metrics = {"nmi": nmi, "acc": acc, "ari": ari, "silhouette": sil, **ilp_metrics}
    return metrics, cluster_labels


# =========================================================================
# Main
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
    parser.add_argument("--apy_15", action="store_true",
                        help="Use DDC paper's 15-class aPY subset (for direct comparison)")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="PCA dimensionality reduction before clustering")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs with different seeds (for mean±std)")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--output_root", type=str, default="results",
                        help="Root dir for outputs; use results_v2 for new runs")
    parser.add_argument("--standardize", action="store_true",
                        help="z-score features before clustering (ablation)")
    args = parser.parse_args()

    setup_logging()
    device = select_device(args.use_gpu)

    # --- Output directory ---
    suffix = f"_k{args.n_clusters}" if args.n_clusters else ""
    suffix += f"_pca{args.pca_dim}" if args.pca_dim else ""
    ds_name = f"{args.dataset}_15" if args.apy_15 else args.dataset
    output_dir = os.path.join(args.output_root, ds_name, f"{args.mode}{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset ---
    paths = get_dataset_paths(args.dataset, use_sample=args.use_sample)
    cfg = DATASET_CONFIGS[args.dataset]

    if args.use_sample:
        create_sample_dataset(
            cfg["source_dir"],
            os.path.join("samples", args.dataset),
            os.path.join(cfg["source_dir"], "classes.txt"),
            sample_size=args.sample_size,
        )

    if not os.path.exists(paths["attr_file"]):
        alt = paths["attr_file"].replace("labels.txt", "AwA2-labels.txt")
        if os.path.exists(alt):
            paths["attr_file"] = alt
        else:
            raise FileNotFoundError(f"Labels file not found: {paths['attr_file']}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Class filter for DDC's 15-class aPY subset
    class_filter = APY_DDC_15_CLASSES if args.apy_15 and args.dataset == "apy" else None

    dataset = AttributeDataset(
        img_dir=paths["img_dir"], attr_file=paths["attr_file"],
        pred_file=paths["pred_file"], classes_file=paths["classes_file"],
        transform=transform, train=True, class_filter=class_filter,
    )

    K = args.n_clusters or len(np.unique(dataset.labels))
    logging.info(f"=== {cfg['name']} | Mode: {args.mode.upper()} | K={K} | "
                 f"N={len(dataset)} | Runs={args.n_runs} ===")

    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4,
        pin_memory=True, collate_fn=custom_collate, shuffle=False,
    )

    # =====================================================================
    # Step 1: Extract features (cached)
    # =====================================================================
    start = time.time()
    cache_tag = f"{args.dataset}{'_15' if args.apy_15 else ''}"
    cache_path = get_cache_path(cache_tag, args.use_sample, args.sample_size)
    features = extract_resnet_features(dataloader, device, cache_path=cache_path)
    true_labels = np.array(dataset.labels)
    symbolic_tags = dataset.symbolic_tags
    logging.info(f"Features: {features.shape} ({time.time()-start:.1f}s)")

    # =====================================================================
    # Step 2-4: Run clustering + evaluation (multi-seed)
    # =====================================================================
    seeds = [42] if args.n_runs == 1 else [42 + i for i in range(args.n_runs)]
    all_metrics = []
    primary_labels = None

    for i, seed in enumerate(seeds):
        logging.info(f"--- Run {i+1}/{len(seeds)} (seed={seed}) ---")
        metrics, cluster_labels = run_single(
            features, true_labels, symbolic_tags, None,
            args.mode, K, seed, cfg, paths, output_dir,
            pca_dim=args.pca_dim, standardize=args.standardize,
        )
        all_metrics.append(metrics)
        if seed == 42:
            primary_labels = cluster_labels

        logging.info(f"  NMI={metrics['nmi']:.4f}  ACC={metrics['acc']:.4f}  "
                     f"ARI={metrics['ari']:.4f}  Sil={metrics['silhouette']:.4f}"
                     + (f"  TC={metrics.get('avg_tc',0):.4f}  "
                        f"ITF={metrics.get('avg_itf',0):.4f}"
                        if 'avg_tc' in metrics else ""))

    # Aggregate results
    metric_keys = ["nmi", "acc", "ari", "silhouette", "avg_tc", "avg_itf"]
    agg = {}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            agg[key] = round(np.mean(vals), 4)
            if len(vals) > 1:
                agg[f"{key}_std"] = round(np.std(vals), 4)

    if args.n_runs > 1:
        logging.info(f"=== Aggregated ({args.n_runs} runs) ===")
        for key in ["nmi", "acc", "ari"]:
            if f"{key}_std" in agg:
                logging.info(f"  {key.upper()}: {agg[key]:.4f} ± {agg[f'{key}_std']:.4f}")

    # =====================================================================
    # Step 5: Save everything (using primary run)
    # =====================================================================
    cluster_labels = primary_labels

    # Cluster attribute summaries
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
              title=f"{cfg['name']} — {args.mode.upper()} (NMI={agg.get('nmi',0):.3f})")
    plot_pca(features, cluster_labels,
             os.path.join(output_dir, "pca.png"),
             title=f"{cfg['name']} — {args.mode.upper()} PCA (NMI={agg.get('nmi',0):.3f})")

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
        "n_runs": args.n_runs,
        "pca_dim": args.pca_dim,
        "metrics": agg,
        "per_run_metrics": all_metrics if args.n_runs > 1 else None,
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

    full_results = {**summary, "clusters": cluster_labels.tolist()}
    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    logging.info(f"Results saved to {output_dir}/")
    logging.info(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()