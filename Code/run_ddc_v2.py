"""Phase 1 runner: paper-faithful DDC on cached ResNet-101 features.

  python run_ddc_v2.py --sanity --use_gpu
  python run_ddc_v2.py --dataset awa2 --use_gpu          # full scale, later steps only

Writes to results_v2/ only. Refuses to write anywhere under results/.
"""

import argparse
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

from utils import setup_logging
from ddc_v2 import data as ddc_data
from ddc_v2.train import DDCTrainer


def build_args():
    p = argparse.ArgumentParser(description="DDC rebuild (Zhang & Davidson, IJCAI 2021)")

    # data
    p.add_argument("--dataset", choices=["awa2", "apy"], default="awa2")
    p.add_argument("--apy_15", action="store_true")
    p.add_argument("--n_clusters", type=int, default=None)
    p.add_argument("--sanity", action="store_true",
                   help="1600-sample subset of the training split")
    p.add_argument("--n_sanity", type=int, default=1600)
    p.add_argument("--preproc", choices=["zscore", "l2", "none"], default="none",
                   help="JUDGMENT CALL — paper is silent on feature preprocessing; "
                        "'none' matches every reported Phase 1 result")

    # tags
    p.add_argument("--tag_ratio", type=float, default=0.5, help="paper's r")
    p.add_argument("--mask_mode", choices=["instance", "entry"], default="instance")
    p.add_argument("--pairs_from", choices=["annotated", "all"], default="annotated")

    # objectives
    p.add_argument("--lam", type=float, default=1.0, help="paper's lambda")
    p.add_argument("--gamma", type=float, default=100.0, help="paper's gamma")
    p.add_argument("--top_l", type=int, default=1, help="paper's l")
    p.add_argument("--alpha", type=int, default=8, help="paper's alpha")
    p.add_argument("--mu", type=float, default=1.0,
                   help="weight on H(Y); 1.0 = Eq (1) as written, >1 = ablation")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--symmetric_kl", action="store_true")
    p.add_argument("--skip_ilp", action="store_true",
                   help="ablation: never solve the ILP, keep g = identity "
                        "(the paper's Fig 4b ablation); also makes runs fast)")

    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=1200)
    p.add_argument("--pretrain_epochs", type=int, default=10)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--epochs_per_round", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--ilp_time_limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # plumbing
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--output_root", type=str, default="results_v2")
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--baseline_nmi", type=float, default=0.869,
                   help="horizontal reference line on the NMI plot")
    p.add_argument("--log_file", type=str, default=None,
                   help="log filename; defaults to ddc_phase1_<run_tag>.log so "
                        "concurrent runs do not interleave into one file")
    return p.parse_args()


def guard_output_root(root):
    parts = os.path.normpath(os.path.abspath(root)).split(os.sep)
    if "results" in parts:
        raise SystemExit(
            f"Refusing to write under a 'results' directory ({root}). "
            f"results/ holds the verified v1 evidence; use results_v2/."
        )


def main():
    args = build_args()
    guard_output_root(args.output_root)
    setup_logging(log_filename=args.log_file
                  or f"ddc_phase1_{args.run_tag or 'default'}.log")

    if not args.skip_ilp:
        try:
            import pulp
        except ImportError:
            raise SystemExit(
                "pulp is not installed in this environment, so the ILP cannot be "
                "solved. Install it (pip install pulp) or pass --skip_ilp."
            )

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- data -------------------------------------------------------------
    ds, paths = ddc_data.build_dataset(args.dataset, apy_15=args.apy_15)
    features = ddc_data.load_cached_features(args.dataset, apy_15=args.apy_15)
    labels = np.asarray(ds.labels)
    if features.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"Feature/label length mismatch ({features.shape[0]} vs {labels.shape[0]}). "
            f"The cache was built from a different split — delete it and re-extract."
        )

    tags = ddc_data.load_binary_tags(args.dataset, ds, paths,
                                     class_filter_used=bool(args.apy_15))

    keep = ddc_data.subsample(features.shape[0],
                              args.n_sanity if args.sanity else None,
                              seed=args.seed)
    features, labels, tags = features[keep], labels[keep], tags[keep]

    K = args.n_clusters or int(len(np.unique(labels)))
    features = ddc_data.preprocess_features(features, args.preproc)
    tags_obs, annotated = ddc_data.apply_annotation_ratio(
        tags, r=args.tag_ratio, mode=args.mask_mode, seed=args.seed)

    logging.info(f"=== DDC rebuild | {args.dataset}{' (sanity)' if args.sanity else ''} "
                 f"| N={features.shape[0]} | K={K} | M={tags.shape[1]} "
                 f"| preproc={args.preproc} | r={args.tag_ratio} ===")

    # --- output dir -------------------------------------------------------
    tag = args.run_tag or (
        f"{'sanity' if args.sanity else 'full'}_k{K}_{args.preproc}"
        f"_lam{args.lam:g}_mu{args.mu:g}_seed{args.seed}"
    )
    output_dir = os.path.join(args.output_root, "ddc_phase1", args.dataset, tag)
    guard_output_root(output_dir)

    # --- train ------------------------------------------------------------
    trainer = DDCTrainer(
        features=features, tags_observed=tags_obs, annotated_mask=annotated,
        true_labels=labels, n_clusters=K, args=args,
        predicate_names=ddc_data.load_predicate_names(paths["predicates_file"]),
        device=device,
    )
    best = trainer.fit()
    summary = trainer.save(output_dir)

    final = summary["final"]
    logging.info("=== RESULT ===")
    logging.info(f"final  NMI={final['nmi']:.4f} ACC={final['acc']:.4f} "
                 f"ARI={final['ari']:.4f} active={final['active_argmax']}/{K}")
    logging.info(f"best   NMI={best.get('nmi')} at step {best.get('step')} "
                 f"({best.get('phase')} round {best.get('round')})")
    logging.info(f"TC={summary['description_metrics']['avg_tc']} "
                 f"ITF={summary['description_metrics']['avg_itf']} (nats)")


if __name__ == "__main__":
    main()