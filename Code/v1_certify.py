"""Recover v1's seed-42 cluster assignments and stage them for certification.

Track A / Step 4. v1 (`main_experiments.py` on `main`) never persisted cluster
assignments -- it writes only ilp_descriptions.json (seed 42), summary.json and
full_results.json, and np.savez appears solely for the feature cache. Without the
assignment vector there is no Q, and without Q the description ILP cannot be
re-solved.

The assignments are recoverable because v1's clustering is deterministic given the
seed and the cached features:

  ddc     -> KMeans(n_clusters=K, random_state=42, n_init=10)     (line 241)
  ddeccs  -> get_base_clusterings -> build_sparse_consensus ->
             SpectralClustering(random_state=42)                   (lines 243-248)

with `feats = features.copy()`, no PCA and no standardisation for the shipped runs
(`summary.json` records pca_dim: null; run_all.sh passes no --standardize).

This is reconstruction, not experimentation: it produces no new result and must
reproduce something already in the thesis. The verification gate is what makes it
safe -- ilp_descriptions.json stores n_members for every cluster of the seed-42
run, so a reconstruction that reproduces all K counts cluster-for-cluster IS the
shipped clustering. The script refuses to write anything if it does not.

Also recomputes v1's own alpha, exactly as v1 does (main_experiments.py:261):
    alpha = max(2, min(8, round(16 * symbolic_tags.mean())))
and stages v1's TAG MATRIX -- dataset.symbolic_tags, the continuous per-class
matrix -- because ilp_probe.py otherwise rebuilds binary tags from
predicate-matrix-binary.txt, which is a different Q and would make the
certification meaningless.

Usage:
    python v1_certify.py --dataset awa2   --mode ddc --log_file v1_certify.log
    python v1_certify.py --dataset apy    --mode ddc --log_file v1_certify.log
    python v1_certify.py --dataset apy    --mode ddc --apy_15 --log_file v1_certify.log

Writes results_v2/v1_certify/<tag>/{assignments.npy,tags.npy,summary.json}.
Never writes under Code/results/.
"""

import argparse
import json
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ddc_v2.data import build_dataset
from utils import setup_logging, clustering_acc, get_base_clusterings, \
    build_sparse_consensus


def guard_output_root(path):
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if "results" in parts:
        raise SystemExit(
            f"Refusing to write under a 'results' directory ({path}). "
            f"Code/results/ is read-only for this task.")


def shipped_dir(dataset, apy_15, mode):
    return os.path.join("results", "apy_15" if apy_15 else dataset, mode)


def reconstruct(feats, K, mode, seed=42):
    """Reproduce v1's clustering. Mirrors main_experiments.run_single lines 239-248."""
    if mode in ("kmeans", "ddc"):
        return KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(feats)
    base_labels = get_base_clusterings(feats, n_clusters=K, seed=seed)
    consensus = build_sparse_consensus(base_labels, feats)
    return SpectralClustering(n_clusters=K, affinity="precomputed",
                              assign_labels="kmeans",
                              random_state=seed).fit_predict(consensus)


def verify(labels, true_labels, K, src):
    """Gate. Returns a dict; raises SystemExit if the reconstruction is not v1's."""
    report = {}

    with open(os.path.join(src, "ilp_descriptions.json")) as f:
        shipped = {c["cluster_id"]: c["n_members"] for c in json.load(f)}
    sizes = np.bincount(labels, minlength=K)
    mismatched = [(k, int(sizes[k]), shipped[k])
                  for k in sorted(shipped) if int(sizes[k]) != shipped[k]]
    report["clusters_checked"] = len(shipped)
    report["per_cluster_exact"] = not mismatched
    if mismatched:
        raise SystemExit(
            f"VERIFICATION FAILED: {len(mismatched)} of {len(shipped)} clusters "
            f"differ in size from the shipped run (first: {mismatched[:5]}). "
            f"This is not v1's clustering; refusing to certify against it.")
    logging.info(f"VERIFIED: all {len(shipped)} cluster sizes match "
                 f"ilp_descriptions.json cluster-for-cluster")

    with open(os.path.join(src, "summary.json")) as f:
        summary = json.load(f)
    cs = summary.get("cluster_sizes", {})
    ours = {"min": int(sizes.min()), "max": int(sizes.max()),
            "mean": round(float(sizes.mean()), 1)}
    report["cluster_sizes_shipped"] = cs
    report["cluster_sizes_ours"] = ours
    report["cluster_sizes_match"] = all(
        abs(float(cs[k]) - float(ours[k])) < 0.05 for k in ("min", "max", "mean")
        if k in cs)
    logging.info(f"cluster_sizes shipped {cs} vs ours {ours} -> "
                 f"{'match' if report['cluster_sizes_match'] else 'MISMATCH'}")

    with open(os.path.join(src, "full_results.json")) as f:
        per_run = json.load(f).get("per_run_metrics", [])
    ours_m = {"nmi": round(float(normalized_mutual_info_score(true_labels, labels)), 4),
              "acc": round(float(clustering_acc(true_labels, labels)), 4),
              "ari": round(float(adjusted_rand_score(true_labels, labels)), 4)}
    ship_m = {k: round(float(per_run[0][k]), 4) for k in ("nmi", "acc", "ari")
              if per_run and k in per_run[0]} if per_run else {}
    report["metrics_ours"] = ours_m
    report["metrics_shipped_seed42"] = ship_m
    report["metrics_match"] = bool(ship_m) and all(
        abs(ours_m[k] - ship_m[k]) < 5e-4 for k in ship_m)
    logging.info(f"seed-42 metrics shipped {ship_m} vs ours {ours_m} -> "
                 f"{'match' if report['metrics_match'] else 'MISMATCH'}")
    if ship_m and not report["metrics_match"]:
        logging.warning("Cluster sizes matched but metrics did not — report this "
                        "rather than certifying on it.")
    return report


def main():
    ap = argparse.ArgumentParser(description="Recover v1 seed-42 assignments")
    ap.add_argument("--dataset", choices=["awa2", "apy"], required=True)
    ap.add_argument("--apy_15", action="store_true")
    ap.add_argument("--mode", choices=["ddc", "ddeccs"], default="ddc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--output_root", default="results_v2")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    guard_output_root(args.output_root)
    setup_logging(log_filename=args.log_file or "v1_certify.log")

    tag = f"{'apy_15' if args.apy_15 else args.dataset}_{args.mode}_s{args.seed}"
    logging.info(f"=== v1 reconstruction | {tag} ===")

    # build_dataset carries the AwA2-labels.txt fallback that the direct
    # AttributeDataset construction in main_experiments.py lacks; otherwise the
    # arguments are identical (transform=None only affects image loading, which
    # this script never does).
    ds, paths = build_dataset(args.dataset, apy_15=args.apy_15)
    true_labels = np.array(ds.labels)
    tags = ds.symbolic_tags
    K = args.n_clusters or int(len(np.unique(ds.labels)))

    cache = os.path.join("cache", f"resnet101_{args.dataset}"
                                  f"{'_15' if args.apy_15 else ''}_full.npz")
    with np.load(cache) as z:
        feats = z["features"].copy()
    if feats.shape[0] != len(ds):
        raise SystemExit(f"Feature cache {cache} has {feats.shape[0]} rows but the "
                         f"dataset has {len(ds)}; refusing to proceed.")
    logging.info(f"features {feats.shape} from {cache}; tags {tags.shape} "
                 f"(mean {tags.mean():.4f}); K={K}")

    labels = reconstruct(feats, K, args.mode, seed=args.seed)

    src = shipped_dir(args.dataset, args.apy_15, args.mode)
    report = verify(labels, true_labels, K, src)

    # v1's own alpha rule, main_experiments.py:261
    alpha = max(2, min(8, round(16 * float(tags.mean()))))
    logging.info(f"v1 alpha rule: density {tags.mean():.3f} -> alpha={alpha}")

    out_dir = os.path.join(args.output_root, "v1_certify", tag)
    guard_output_root(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "assignments.npy"), labels)
    np.save(os.path.join(out_dir, "tags.npy"), tags)

    summary = {
        "n_clusters": K,
        "config": {
            "dataset": args.dataset, "apy_15": bool(args.apy_15),
            "mode": args.mode, "seed": args.seed, "alpha": alpha,
            "tag_ratio": 1.0, "mask_mode": "none", "sanity": False,
            "ilp_time_limit": 30, "source": "v1 reconstruction",
        },
        "ilp_last": {"beta": None, "n_tags": int(tags.shape[1])},
        "description_metrics": {"avg_tc": None, "avg_itf": None},
        "verification": report,
        "shipped_dir": src,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote {out_dir}/ (assignments.npy, tags.npy, summary.json)")


if __name__ == "__main__":
    main()