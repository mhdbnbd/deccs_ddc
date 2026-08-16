"""Recover aPY's per-object binary attributes and align them to the processed dataset.

Track A / Step 3, Task 2. `setup_apy.py` averages each object's 64 binary
attributes into per-class means and keeps only those, which is why the shipped
pipeline treats aPY exactly like AwA2. The per-object vectors are still
recoverable: every processed file is named `<source-stem>_obj<NNNNN>.jpg`, where
NNNNN is a global counter that `setup_apy.py` increments once per successfully
processed annotation entry, in annotation-file order. Replaying that order
recovers the mapping.

The reconstruction is verified three ways before anything is written:

  1. the object indices parsed out of labels.txt must be exactly 0..N-1
  2. every labels.txt record must match an annotation entry, in order, on both
     class name and source stem
  3. the per-class means of the recovered vectors must reproduce
     predicate-matrix-continuous.txt to 1e-6 -- that file is precisely what
     setup_apy.py wrote from these vectors, so a single misassigned row breaks it

Check 3 is the decisive one: it is an independent artefact written by the
original run, not something this script can influence.

Nothing under Code/results/ is touched, no existing dataset file is modified,
and setup_apy.py keeps producing exactly what it produces today.

Usage:
    python build_apy_instance_tags.py --log_file apy_instance_tags.log
"""

import argparse
import json
import logging
import math
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

from utils import setup_logging
from dataset import AttributeDataset, get_dataset_paths, APY_DDC_15_CLASSES
from ddc_v2.data import apply_annotation_ratio

ANNOTATION_FILES = ["apascal_train.txt", "apascal_test.txt", "ayahoo_test.txt"]
OBJ_RE = re.compile(r"^(?P<stem>.+)_obj(?P<idx>\d+)\.jpg$")


def guard_output_root(path):
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if "results" in parts:
        raise SystemExit(f"Refusing to write under a 'results' directory ({path}).")


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def parse_annotation_files(ann_dir):
    """Same parse and same file order as setup_apy.py."""
    entries = []
    for fname in ANNOTATION_FILES:
        path = os.path.join(ann_dir, fname)
        if not os.path.exists(path):
            logging.warning(f"{fname} not present in {ann_dir} - skipped")
            continue
        n0 = len(entries)
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 70:
                    continue
                entries.append((parts[0], parts[1],
                                np.array([int(x) for x in parts[6:70]], dtype=np.int8)))
        logging.info(f"{fname}: {len(entries) - n0} objects")
    if not entries:
        raise SystemExit(f"No annotation entries parsed from {ann_dir}")
    logging.info(f"Total annotation entries: {len(entries)}")
    return entries


def read_label_records(dataset_dir):
    """labels.txt lines are '<class>/<stem>_objNNNNN.jpg'."""
    path = os.path.join(dataset_dir, "labels.txt")
    records = []
    with open(path) as f:
        for line in f:
            rel = line.strip().split()[0] if line.strip() else ""
            if not rel:
                continue
            cls, out_name = rel.split("/", 1)
            m = OBJ_RE.match(out_name)
            if m is None:
                raise SystemExit(
                    f"labels.txt entry does not carry an object index: {rel}. "
                    f"This dataset was not produced by the current setup_apy.py.")
            records.append({"idx": int(m.group("idx")), "cls": cls,
                            "stem": m.group("stem"), "out_name": out_name})
    logging.info(f"labels.txt: {len(records)} instances")

    idx = sorted(r["idx"] for r in records)
    if idx != list(range(len(records))):
        missing = sorted(set(range(len(records))) - set(idx))[:10]
        raise SystemExit(
            f"Object indices are not contiguous 0..{len(records) - 1} "
            f"(first gaps: {missing}). Refusing to reconstruct.")
    records.sort(key=lambda r: r["idx"])
    return records


def reconstruct(entries, records):
    """Walk annotation entries in order, consuming one per labels.txt record."""
    mapping, p = {}, 0
    for rec in records:
        while p < len(entries):
            img_name, cls, attrs = entries[p]
            if cls == rec["cls"] and os.path.splitext(img_name)[0] == rec["stem"]:
                mapping[rec["out_name"]] = attrs
                p += 1
                break
            p += 1
        else:
            raise SystemExit(
                f"Ran out of annotation entries at object {rec['idx']} "
                f"({rec['cls']}/{rec['stem']}). Reconstruction failed.")
    skipped = len(entries) - len(records)
    logging.info(f"Matched {len(mapping)}/{len(records)} instances; "
                 f"{skipped} annotation entries had no processed image")

    # The class-mean check below cannot see a permutation *within* one class.
    # The only way that could arise is two objects of the same class annotated in
    # the same source image: they share class and stem, so they are told apart
    # only by their order in the annotation file. Both setup_apy.py and this
    # matcher iterate those lines in the same order, so the assignment is
    # preserved by construction -- but quantify the exposure rather than assume.
    from collections import Counter
    groups = Counter((r["cls"], r["stem"]) for r in records)
    ambiguous = {k: v for k, v in groups.items() if v > 1}
    n_ambig = int(sum(ambiguous.values()))
    logging.info(f"Order-dependent assignments: {len(ambiguous)} source images "
                 f"hold two or more objects of the same class, covering "
                 f"{n_ambig} instances ({100.0 * n_ambig / len(records):.1f}%)")
    return mapping, {"groups": len(ambiguous), "instances": n_ambig}


def verify_against_class_means(mapping, records, dataset_dir, atol=1e-6):
    """The decisive check: per-class means must reproduce the shipped matrix."""
    classes_file = os.path.join(dataset_dir, "classes.txt")
    order = []
    with open(classes_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                order.append(parts[1])

    shipped = np.loadtxt(os.path.join(dataset_dir,
                                      "predicate-matrix-continuous.txt"))
    if shipped.shape[0] != len(order):
        raise SystemExit(f"predicate matrix has {shipped.shape[0]} rows but "
                         f"classes.txt lists {len(order)} classes")

    by_class = {}
    for rec in records:
        by_class.setdefault(rec["cls"], []).append(mapping[rec["out_name"]])

    rebuilt = np.zeros_like(shipped, dtype=np.float64)
    for i, cls in enumerate(order):
        if cls in by_class:
            rebuilt[i] = np.mean(np.stack(by_class[cls]).astype(np.float64), axis=0)

    diff = np.abs(rebuilt - shipped)
    worst = float(diff.max())
    if worst > atol:
        bad = int(np.unravel_index(diff.argmax(), diff.shape)[0])
        raise SystemExit(
            f"VERIFICATION FAILED: reconstructed per-class means differ from "
            f"predicate-matrix-continuous.txt by up to {worst:.3e} "
            f"(worst row {bad} = class '{order[bad]}'). The per-object mapping "
            f"is wrong; refusing to write anything.")
    logging.info(f"VERIFIED: per-class means reproduce "
                 f"predicate-matrix-continuous.txt (max abs diff {worst:.2e})")
    return worst


def write_sidecar(mapping, records, path, force=False):
    if os.path.exists(path) and not force:
        raise SystemExit(f"{path} already exists; pass --force to overwrite.")
    with open(path, "w") as f:
        f.write("# aPY per-object binary attributes recovered from the raw "
                "annotations by build_apy_instance_tags.py\n")
        f.write("# <processed filename>\\t<64 space-separated 0/1 values>\n")
        for rec in records:
            vals = " ".join(str(int(v)) for v in mapping[rec["out_name"]])
            f.write(f"{rec['out_name']}\t{vals}\n")
    logging.info(f"Wrote {path} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Characterisation on the 15-class train split
# ---------------------------------------------------------------------------

def zero_distance_fraction(tags, labels):
    """Fraction of within-class instance pairs whose tag vectors are identical."""
    same, total = 0, 0
    for c in np.unique(labels):
        rows = tags[labels == c]
        n = rows.shape[0]
        if n < 2:
            continue
        total += n * (n - 1) // 2
        _, counts = np.unique(rows, axis=0, return_counts=True)
        same += int(sum(k * (k - 1) // 2 for k in counts))
    return (same / total if total else None), total


def characterise(mapping, paths, args):
    ds = AttributeDataset(
        img_dir=paths["img_dir"], attr_file=paths["attr_file"],
        pred_file=paths["pred_file"], classes_file=paths["classes_file"],
        transform=None, train=True, class_filter=APY_DDC_15_CLASSES,
    )
    names = [os.path.basename(p) for p in ds.image_paths]
    missing = [n for n in names if n not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} split instances have no recovered tag "
                         f"vector (first: {missing[:3]}). Refusing to report.")

    tags = np.stack([mapping[n] for n in names]).astype(np.float32)
    labels = np.asarray(ds.labels)
    N, M = tags.shape
    K = int(len(np.unique(labels)))
    logging.info(f"15-class train split: N={N}, K={K}, M={M}")

    stats = {"n_instances": N, "n_classes": K, "n_attributes": M}

    # cached features -- the alignment guard the runner will also apply
    if os.path.exists(args.feature_cache):
        with np.load(args.feature_cache) as z:
            keys = list(z.keys())
            shapes = {k: list(np.shape(z[k])) for k in keys}
            n_feat = int(np.shape(z["features"])[0]) if "features" in z else None
        stats["feature_cache"] = {"path": args.feature_cache, "arrays": shapes,
                                  "matches_split": n_feat == N}
        if n_feat == N:
            logging.info(f"Feature cache {args.feature_cache}: {shapes} - "
                         f"length matches the split ({N})")
        else:
            logging.error(f"Feature cache {args.feature_cache} holds {n_feat} rows "
                          f"but the 15-class train split has {N}. Do NOT run "
                          f"training against this cache.")
    else:
        stats["feature_cache"] = {"path": args.feature_cache, "arrays": None,
                                  "matches_split": None}
        logging.warning(f"No feature cache at {args.feature_cache}")

    # density
    per_instance = tags.sum(axis=1)
    stats["density_per_instance"] = round(float(tags.mean()), 4)
    stats["active_attrs_per_instance"] = {
        "mean": round(float(per_instance.mean()), 2),
        "min": int(per_instance.min()), "max": int(per_instance.max()),
    }
    shipped = np.loadtxt(paths["pred_file"])
    cls_rows = sorted({int(l) for l in labels})
    stats["density_per_class_mean_matrix"] = round(float(shipped.mean()), 4)
    logging.info(f"per-instance binary density {stats['density_per_instance']} "
                 f"({stats['active_attrs_per_instance']['mean']} of {M} attrs per "
                 f"instance); per-class-mean matrix density "
                 f"{stats['density_per_class_mean_matrix']}")

    # distinct rows and within-class degeneracy, raw
    stats["distinct_rows_raw"] = int(np.unique(tags, axis=0).shape[0])
    frac, n_pairs = zero_distance_fraction(tags, labels)
    stats["within_class_zero_distance_fraction_raw"] = (round(frac, 4)
                                                        if frac is not None else None)
    stats["within_class_pairs"] = n_pairs
    logging.info(f"distinct tag rows (raw): {stats['distinct_rows_raw']} of {N}; "
                 f"within-class exact-zero distance fraction "
                 f"{stats['within_class_zero_distance_fraction_raw']} "
                 f"over {n_pairs} pairs")

    # after r-masking, per seed -- this is what Eq (5) actually sees
    per_seed = []
    for seed in args.seeds:
        obs, ann = apply_annotation_ratio(tags, r=args.tag_ratio,
                                          mode="instance", seed=seed)
        frac_all, _ = zero_distance_fraction(obs, labels)
        frac_ann, _ = zero_distance_fraction(obs[ann], labels[ann])
        Q = np.stack([obs[labels == c].mean(axis=0) for c in np.unique(labels)])
        cov = Q.sum(axis=1)
        per_seed.append({
            "seed": seed,
            "annotated": int(ann.sum()),
            "distinct_rows_masked": int(np.unique(obs, axis=0).shape[0]),
            "zero_distance_fraction_masked": round(float(frac_all), 4),
            "zero_distance_fraction_annotated_only": round(float(frac_ann), 4),
            "max_coverage_min_class": round(float(cov.min()), 2),
            "max_coverage_mean_class": round(float(cov.mean()), 2),
        })
        logging.info(f"[seed {seed}] annotated {int(ann.sum())}/{N}, distinct rows "
                     f"{per_seed[-1]['distinct_rows_masked']}, zero-distance "
                     f"{per_seed[-1]['zero_distance_fraction_masked']} (all) / "
                     f"{per_seed[-1]['zero_distance_fraction_annotated_only']} "
                     f"(annotated only), min class coverage "
                     f"{per_seed[-1]['max_coverage_min_class']}")
    stats["per_seed"] = per_seed

    # alpha feasibility. Eq (3) needs sum_j W_ij Q_ij >= alpha with W binary, so
    # the largest coverage any cluster can reach is its full row sum of Q.
    worst = min(s["max_coverage_min_class"] for s in per_seed)
    stats["alpha"] = {
        "requested": args.alpha,
        "min_class_max_coverage": worst,
        "feasible_on_true_partition": bool(worst >= args.alpha),
        "largest_feasible_alpha": int(math.floor(worst)),
        "note": "necessary condition evaluated on the ground-truth partition; "
                "learned clusters are less pure so this is an upper bound",
    }
    stats["beta_lower_bound_arithmetic"] = int(math.ceil(K * args.alpha / M))
    logging.info(f"alpha={args.alpha} feasible on the true partition: "
                 f"{stats['alpha']['feasible_on_true_partition']} "
                 f"(worst class reaches {worst}; largest feasible alpha "
                 f"{stats['alpha']['largest_feasible_alpha']}); "
                 f"beta >= {stats['beta_lower_bound_arithmetic']} by arithmetic")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Recover aPY per-instance tags")
    ap.add_argument("--annotation_dir", default="data/aPY-data/attribute_data")
    ap.add_argument("--dataset_dir", default="data/aPY-data/aPY")
    ap.add_argument("--output", default=None,
                    help="default <dataset_dir>/per-instance-attributes.tsv")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--tag_ratio", type=float, default=0.5)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--feature_cache", default="cache/resnet101_apy_15_full.npz")
    ap.add_argument("--stats_out", default="results_v2/apy15_tag_stats.json")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output = args.output or os.path.join(args.dataset_dir,
                                              "per-instance-attributes.tsv")
    guard_output_root(args.stats_out)
    setup_logging(log_filename=args.log_file or "apy_instance_tags.log")

    logging.info(f"=== aPY per-instance tag reconstruction | "
                 f"{args.annotation_dir} -> {args.output} ===")

    entries = parse_annotation_files(args.annotation_dir)
    records = read_label_records(args.dataset_dir)
    mapping, ambiguity = reconstruct(entries, records)
    worst = verify_against_class_means(mapping, records, args.dataset_dir)
    write_sidecar(mapping, records, args.output, force=args.force)

    paths = get_dataset_paths("apy")
    stats = characterise(mapping, paths, args)
    stats["verification_max_abs_diff"] = worst
    stats["sidecar"] = args.output
    stats["n_annotation_entries"] = len(entries)
    stats["n_processed_instances"] = len(records)
    stats["order_dependent_assignments"] = ambiguity

    os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Wrote {args.stats_out}")


if __name__ == "__main__":
    main()