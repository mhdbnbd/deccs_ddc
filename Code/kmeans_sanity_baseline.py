"""Matched baselines on the exact sanity subset.

The 0.869 figure is k-means on 29,857 samples. Comparing a 1,600-sample DDC run
against it is not a like-for-like test, so this script computes the baselines on
the same subset, with the same seed and the same preprocessing:

  1. k-means on the features alone  (your v1 baseline, at this N)
  2. k-means on features concatenated with the observed tags at r=0.5, which is
     the form of the k-means row in the DDC paper's Table 3

Prints only — writes nothing to disk.

  python kmeans_sanity_baseline.py --use_subset
  python kmeans_sanity_baseline.py                 # full training split
"""

import argparse
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from utils import clustering_acc
from ddc_v2 import data as ddc_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["awa2", "apy"], default="awa2")
    p.add_argument("--apy_15", action="store_true")
    p.add_argument("--per_instance_tags", action="store_true",
                   help="aPY only: true per-instance binary tags")
    p.add_argument("--per_instance_tags_path", type=str,
                   default="data/aPY-data/aPY/per-instance-attributes.tsv")
    p.add_argument("--use_subset", action="store_true",
                   help="use the same subset run_ddc_v2.py --sanity uses")
    p.add_argument("--n_sanity", type=int, default=1600)
    p.add_argument("--n_clusters", type=int, default=None)
    p.add_argument("--preproc", choices=["zscore", "l2", "none"], default="none")
    p.add_argument("--tag_ratio", type=float, default=0.5)
    p.add_argument("--mask_mode", choices=["instance", "entry"], default="instance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_runs", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ds, paths = ddc_data.build_dataset(args.dataset, apy_15=args.apy_15)
    features = ddc_data.load_cached_features(args.dataset, apy_15=args.apy_15)
    labels = np.asarray(ds.labels)
    if features.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"Feature/label length mismatch ({features.shape[0]} vs {labels.shape[0]}). "
            f"The cache was built from a different split — delete it and re-extract."
        )
    tags = ddc_data.load_binary_tags(
        args.dataset, ds, paths, class_filter_used=bool(args.apy_15),
        per_instance_path=(args.per_instance_tags_path
                           if args.per_instance_tags else None))

    keep = ddc_data.subsample(features.shape[0],
                              args.n_sanity if args.use_subset else None,
                              seed=args.seed)
    features, labels, tags = features[keep], labels[keep], tags[keep]

    K = args.n_clusters or int(len(np.unique(labels)))
    X = ddc_data.preprocess_features(features, args.preproc)
    tags_obs, _ = ddc_data.apply_annotation_ratio(
        tags, r=args.tag_ratio, mode=args.mask_mode, seed=args.seed)

    # DDC's Table 3 k-means row concatenates the image features with the tags.
    # The paper says only "naively concatenates" and does not state whether the
    # tag block is rescaled, so report both readings:
    #   X_cat      standardised tag block (our judgment call)
    #   X_cat_raw  literal concatenation of the observed 0/1 tag matrix
    T = StandardScaler().fit_transform(tags_obs)
    X_cat = np.hstack([X, T]).astype(np.float32)
    X_cat_raw = np.hstack([X, tags_obs]).astype(np.float32)

    print(f"N={X.shape[0]} K={K} D={X.shape[1]} M={tags.shape[1]} "
          f"preproc={args.preproc} r={args.tag_ratio}")

    for name, data in (("features only", X),
                       ("feats + tags (std)", X_cat),
                       ("feats + tags (raw)", X_cat_raw),
                       ("tags only (raw)", tags_obs.astype(np.float32))):
        rows = []
        for i in range(args.n_runs):
            seed = args.seed + i
            pred = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(data)
            rows.append((
                normalized_mutual_info_score(labels, pred),
                clustering_acc(labels, pred),
                adjusted_rand_score(labels, pred),
                len(np.unique(pred)),
            ))
        arr = np.array(rows, dtype=float)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        print(f"{name:>16}: NMI={m[0]:.4f}+-{s[0]:.4f}  ACC={m[1]:.4f}+-{s[1]:.4f}  "
              f"ARI={m[2]:.4f}+-{s[2]:.4f}  active={m[3]:.0f}/{K}")

    # Ceiling on ACC if only k_act clusters are ever populated.
    sizes = np.sort(np.bincount(labels, minlength=labels.max() + 1))[::-1]
    sizes = sizes[sizes > 0]
    print("\nACC ceiling if only k_act clusters are populated "
          "(sum of the k_act largest class shares):")
    for k_act in (15, 19, 24, 30, 40, K):
        if k_act <= len(sizes):
            print(f"  k_act={k_act:2d} -> {sizes[:k_act].sum() / sizes.sum():.4f}")


if __name__ == "__main__":
    main()