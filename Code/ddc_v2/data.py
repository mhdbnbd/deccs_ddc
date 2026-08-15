"""Data plumbing for the DDC rebuild.

Reuses the existing cached ResNet-101 features (cache/resnet101_*.npz) and the
existing AttributeDataset for labels. Adds two things the shipped pipeline does
not have and the paper requires:

  1. binary per-instance tags (paper section 3.3 defines each t_i as a binary
     vector of length M)
  2. the annotated ratio r, with missing tags mean-imputed (paper section 3.3
     states tags may be missing with probability r and that missing entries are
     mean-imputed; section 4.1 sets r = 0.5 by default)

Nothing here writes to disk.
"""

import logging
import os

import numpy as np

from dataset import AttributeDataset, get_dataset_paths, DATASET_CONFIGS, APY_DDC_15_CLASSES


# ---------------------------------------------------------------------------
# Cached features
# ---------------------------------------------------------------------------

def cache_path_for(dataset_name, apy_15=False, use_sample=False, sample_size=2000):
    """Mirror of main_experiments.get_cache_path so we hit the same files."""
    tag = f"{dataset_name}{'_15' if apy_15 else ''}"
    tag = f"{tag}_{'sample' + str(sample_size) if use_sample else 'full'}"
    return os.path.join("cache", f"resnet101_{tag}.npz")


def load_cached_features(dataset_name, apy_15=False, use_sample=False, sample_size=2000):
    path = cache_path_for(dataset_name, apy_15, use_sample, sample_size)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached features at {path}. Extract them once with the shipped "
            f"pipeline (e.g. `python main_experiments.py --dataset {dataset_name} "
            f"--mode kmeans --use_gpu --output_root results_v2`), then re-run this."
        )
    feats = np.load(path)["features"].astype(np.float32)
    logging.info(f"Loaded cached features {feats.shape} from {path}")
    return feats


def build_dataset(dataset_name, apy_15=False, use_sample=False):
    """AttributeDataset without a transform — we never touch the images here."""
    paths = get_dataset_paths(dataset_name, use_sample=use_sample)
    if not os.path.exists(paths["attr_file"]):
        alt = paths["attr_file"].replace("labels.txt", "AwA2-labels.txt")
        if os.path.exists(alt):
            paths["attr_file"] = alt
        else:
            raise FileNotFoundError(f"Labels file not found: {paths['attr_file']}")

    class_filter = APY_DDC_15_CLASSES if (apy_15 and dataset_name == "apy") else None
    ds = AttributeDataset(
        img_dir=paths["img_dir"], attr_file=paths["attr_file"],
        pred_file=paths["pred_file"], classes_file=paths["classes_file"],
        transform=None, train=True, class_filter=class_filter,
    )
    return ds, paths


# ---------------------------------------------------------------------------
# Binary tags
# ---------------------------------------------------------------------------

def load_binary_tags(dataset_name, ds, paths, class_filter_used=False):
    """
    Per-instance binary tag matrix (N, M).

    Preference order:
      1. predicate-matrix-binary.txt in the dataset source dir (AwA2 ships this)
      2. threshold of the continuous predicate matrix at its midpoint

    Note this is still a *per-class* tag vector broadcast to instances, because
    that is what the shipped AwA2/aPY data path provides. The paper's aPY setting
    uses true per-instance tags; that gap is Phase-1-out-of-scope and is recorded
    in the handoff.
    """
    src = DATASET_CONFIGS[dataset_name]["source_dir"]
    binary_path = os.path.join(src, "predicate-matrix-binary.txt")
    labels = np.asarray(ds.labels)

    if os.path.exists(binary_path):
        mat = np.loadtxt(binary_path)
        mat = (mat > 0.5).astype(np.float32)
        source = binary_path
    else:
        cont = np.loadtxt(paths["pred_file"]).astype(np.float64)
        thr = 0.5 * (cont.min() + cont.max())
        mat = (cont > thr).astype(np.float32)
        source = f"{paths['pred_file']} thresholded at {thr:.3f}"
        logging.warning(f"No predicate-matrix-binary.txt; binarised {source}")

    if class_filter_used:
        # AttributeDataset remaps filtered labels to contiguous 0-based ids and
        # rebuilds label_to_tags accordingly; reuse that mapping's row order.
        raise NotImplementedError(
            "Binary tags for the class-filtered subset are not wired yet. "
            "Phase 1 targets full AwA2; add this when aPY-15 comes up."
        )

    # Unfiltered path: AttributeDataset stores 1-based class ids and maps
    # {i+1: predicate_row_i}, so label L -> row L-1.
    rows = labels - 1
    if rows.min() < 0 or rows.max() >= mat.shape[0]:
        raise ValueError(
            f"Label/predicate row mismatch: labels in [{labels.min()},{labels.max()}], "
            f"predicate matrix has {mat.shape[0]} rows.")
    tags = mat[rows].astype(np.float32)
    logging.info(f"Binary tags {tags.shape} from {source} (density={tags.mean():.3f})")
    return tags


def load_predicate_names(predicates_path):
    """Human-readable attribute names (same parsing as main_experiments)."""
    names = []
    if os.path.exists(predicates_path):
        with open(predicates_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    parts = line.split("\t", 1)
                else:
                    parts = line.split(None, 1)
                names.append(parts[1] if len(parts) > 1 else parts[0])
    return names or None


# ---------------------------------------------------------------------------
# Annotated ratio r + mean imputation
# ---------------------------------------------------------------------------

def apply_annotation_ratio(tags, r=0.5, mode="instance", seed=42):
    """
    Paper section 3.3: an instance's tags may be missing with probability r, and
    missing tags are handled by mean imputation. Section 4.1 sets the annotated
    ratio r to 0.5 by default.

    mode="instance" (default, paper-literal): a fixed random (1-r) fraction of
      instances have their whole tag vector missing; those rows are replaced by
      the column means over the annotated rows.
    mode="entry": each (instance, tag) entry is missing independently with
      probability (1-r); missing entries are replaced by that tag's mean over its
      observed entries.

    The mask is drawn ONCE and held fixed for the whole run. Re-drawing it per
    batch (as the archived trainer did) turns identical same-class tag vectors
    into random ones and destroys the constraint signal.

    Returns (tags_observed, annotated_mask) where annotated_mask is (N,) bool for
    mode="instance" and all-True for mode="entry".
    """
    rng = np.random.default_rng(seed)
    tags = tags.astype(np.float32).copy()
    N, M = tags.shape

    if mode == "instance":
        annotated = rng.random(N) < r
        if annotated.sum() == 0:
            raise ValueError("r too small: no annotated instances")
        col_mean = tags[annotated].mean(axis=0)
        out = tags.copy()
        out[~annotated] = col_mean
        logging.info(f"Annotation r={r} (instance-level): {int(annotated.sum())}/{N} "
                     f"annotated, {N - int(annotated.sum())} mean-imputed")
        return out, annotated

    if mode == "entry":
        observed = rng.random((N, M)) < r
        out = tags.copy()
        for j in range(M):
            obs = observed[:, j]
            fill = tags[obs, j].mean() if obs.any() else tags[:, j].mean()
            out[~obs, j] = fill
        logging.info(f"Annotation r={r} (entry-level): {observed.mean():.3f} observed")
        return out, np.ones(N, dtype=bool)

    raise ValueError(f"unknown mask mode {mode}")


# ---------------------------------------------------------------------------
# Preprocessing and subsetting
# ---------------------------------------------------------------------------

def preprocess_features(features, mode="none"):
    """
    The paper does not state any preprocessing of the ResNet-101 features.
    JUDGMENT CALL — default none (raw activations), which is what every reported
    Phase 1 result uses. An earlier note here claimed raw non-negative ResNet
    activations collapse the MI objective to a single cluster within one epoch;
    the seeded raw runs do not reproduce that (19 active clusters at mu=1,
    48-50 at mu=1.5), so the claim is withdrawn.
    """
    x = features.astype(np.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-8
        return (x - mu) / sd
    if mode == "l2":
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    raise ValueError(f"unknown preprocessing {mode}")


def subsample(n_total, n_keep, seed=42):
    """Fixed random subset of row indices (sanity mode)."""
    if n_keep is None or n_keep >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))