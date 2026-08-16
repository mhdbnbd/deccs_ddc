"""Offline re-solve of the DDC description ILP from a finished run's assignments.

Task 1 of Track A / Step 3. The reportable TC/ITF/beta/tags of a run come from
`trainer.last_W` — the ILP of the FINAL round only (see DDCTrainer.save). That
solve is a deterministic function of `assignments.npy` and the masked tag matrix,
both of which are reproducible from the run directory and the seed. So the
question "did the 20 s limit distort the description metrics?" can be answered
exactly, on CPU, without retraining.

For each beta from 1 upward this reports:
  - the arithmetic necessary condition  K_act * alpha <= beta * M
  - the LP relaxation (infeasible relaxation => the ILP is infeasible, certified
    in milliseconds; feasible relaxation => nothing is settled)
  - the ILP itself, with CBC's real termination status, distinguishing
    proven-optimal / time-limited-incumbent / time-limited-no-solution /
    proven-infeasible
  - TC, ITF and tags kept for every beta that yields a verified solution

Usage:
    python ilp_probe.py --run_dir results_v2/ddc_phase1/awa2/full_mu15_a3_ilp_s42 \
        --time_limit 120 --log_file ilp_probe_s42_t120.log

Reads only. Writes one JSON per run under results_v2/ilp_probe/.
"""

import argparse
import json
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

from utils import setup_logging
from ddc_v2 import data as ddc_data
from ddc_v2.objectives import description_metrics


def guard_output_root(root):
    parts = os.path.normpath(os.path.abspath(root)).split(os.sep)
    if "results" in parts:
        raise SystemExit(
            f"Refusing to write under a 'results' directory ({root}). "
            f"results/ holds the verified v1 evidence; use results_v2/."
        )


def rebuild_tags(cfg):
    """Reproduce the exact masked tag matrix the run trained on."""
    dataset = cfg.get("dataset", "awa2")
    apy_15 = bool(cfg.get("apy_15", False))
    ds, paths = ddc_data.build_dataset(dataset, apy_15=apy_15)
    tags = ddc_data.load_binary_tags(dataset, ds, paths, class_filter_used=apy_15)

    keep = ddc_data.subsample(tags.shape[0],
                              cfg.get("n_sanity") if cfg.get("sanity") else None,
                              seed=cfg["seed"])
    tags = tags[keep]
    tags_obs, _ = ddc_data.apply_annotation_ratio(
        tags, r=cfg.get("tag_ratio", 0.5),
        mode=cfg.get("mask_mode", "instance"), seed=cfg["seed"])
    return tags_obs, ddc_data.load_predicate_names(paths["predicates_file"])


def scan_beta(assignments, tags, n_clusters, alpha, time_limit, beta_max=None,
              threads=1, beta_min=1):
    """Full beta scan with LP relaxation pre-check. Stops at the first beta that
    yields a solution verified against Eq (3) and Eq (4)."""
    import pulp
    import time as _time

    M = tags.shape[1]
    active = [k for k in range(n_clusters) if (assignments == k).sum() > 0]
    K_act = len(active)
    Q = np.stack([tags[assignments == k].mean(axis=0) for k in active]).astype(np.float64)

    lower_bound = int(np.ceil(K_act * alpha / M))
    logging.info(f"K_act={K_act}, M={M}, alpha={alpha}: summing Eq (3) over clusters "
                 f"and Eq (4) over tags gives K_act*alpha <= beta*M, so beta >= "
                 f"{lower_bound} is a necessary condition")

    def build(cat):
        p = pulp.LpProblem("DDC_description", pulp.LpMinimize)
        W = {(i, j): pulp.LpVariable(f"W_{i}_{j}", lowBound=0, upBound=1, cat=cat)
             for i in range(K_act) for j in range(M)}
        p += pulp.lpSum(W.values())
        for i in range(K_act):
            p += pulp.lpSum(W[i, j] * Q[i, j] for j in range(M)) >= alpha
        for j in range(M):
            p += pulp.lpSum(W[i, j] * Q[i, j] for i in range(K_act)) <= beta
        return p, W

    trace, accepted = [], None
    for beta in range(max(1, int(beta_min)), (beta_max or K_act) + 1):
        entry = {"beta": beta}

        if beta < lower_bound:
            entry.update(outcome="infeasible_by_arithmetic", seconds=0.0)
            trace.append(entry)
            logging.info(f"[beta={beta}] infeasible by the summed bound")
            continue

        p, _ = build("Continuous")
        t0 = _time.time()
        p.solve(pulp.PULP_CBC_CMD(msg=0))
        entry["lp_relaxation"] = {
            "status": pulp.LpStatus.get(p.status, "unknown"),
            "objective": (round(float(pulp.value(p.objective)), 2)
                          if p.status == pulp.LpStatusOptimal else None),
            "seconds": round(_time.time() - t0, 2),
        }
        if p.status == pulp.LpStatusInfeasible:
            entry.update(outcome="proven_infeasible_by_lp", seconds=0.0)
            trace.append(entry)
            logging.info(f"[beta={beta}] LP relaxation infeasible -> ILP infeasible")
            continue

        p, W = build("Binary")
        t0 = _time.time()
        p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=threads))
        secs = _time.time() - t0
        sol_code = getattr(p, "sol_status", None)
        if p.status == pulp.LpStatusOptimal:
            outcome = ("proven_optimal" if sol_code == pulp.LpSolutionOptimal
                       else "time_limit_incumbent")
        elif p.status == pulp.LpStatusInfeasible:
            outcome = "proven_infeasible"
        else:
            outcome = "time_limit_no_solution"
        entry.update(outcome=outcome, seconds=round(secs, 1),
                     lp_status=pulp.LpStatus.get(p.status, "unknown"),
                     sol_status=pulp.LpSolution.get(sol_code, "unknown"),
                     objective=(round(float(pulp.value(p.objective)), 2)
                                if p.status == pulp.LpStatusOptimal else None))

        if p.status != pulp.LpStatusOptimal:
            trace.append(entry)
            logging.info(f"[beta={beta}] {outcome} after {secs:.1f}s "
                         f"(limit {time_limit}s)")
            continue

        W_np = np.zeros((K_act, M), dtype=np.float32)
        for i in range(K_act):
            for j in range(M):
                W_np[i, j] = W[i, j].varValue or 0.0
        W_np = (W_np > 0.5).astype(np.float32)
        cover, load = (W_np * Q).sum(axis=1), (W_np * Q).sum(axis=0)
        entry["constraints_verified"] = bool(
            (cover >= alpha - 1e-6).all() and (load <= beta + 1e-6).all())
        entry["n_tags"] = int((W_np.sum(axis=0) > 0.5).sum())
        entry["tags_per_cluster"] = round(float(W_np.sum(axis=1).mean()), 2)
        trace.append(entry)
        logging.info(f"[beta={beta}] {outcome} after {secs:.1f}s, "
                     f"{entry['n_tags']}/{M} tags kept, "
                     f"verified={entry['constraints_verified']}")

        if entry["constraints_verified"]:
            accepted = (beta, W_np, active, entry)
            break

    return trace, accepted, K_act, lower_bound


def main():
    ap = argparse.ArgumentParser(description="Offline ILP re-solve probe")
    ap.add_argument("--run_dir", required=True,
                    help="a finished run directory containing summary.json and "
                         "assignments.npy")
    ap.add_argument("--time_limit", type=int, default=120)
    ap.add_argument("--threads", type=int, default=1,
                    help="CBC threads; >1 can change which optimum is returned, "
                         "so keep 1 for reportable numbers")
    ap.add_argument("--beta_max", type=int, default=8)
    ap.add_argument("--beta_min", type=int, default=1,
                    help="start the scan here; use 4 on AwA2 to pin the smallest "
                         "beta whose solve CBC certifies")
    ap.add_argument("--output_root", type=str, default="results_v2")
    ap.add_argument("--log_file", type=str, default=None)
    args = ap.parse_args()

    guard_output_root(args.output_root)
    setup_logging(log_filename=args.log_file or "ilp_probe.log")

    with open(os.path.join(args.run_dir, "summary.json")) as f:
        summary = json.load(f)
    cfg = summary["config"]
    assignments = np.load(os.path.join(args.run_dir, "assignments.npy"))

    logging.info(f"=== ILP probe | {args.run_dir} | limit {args.time_limit}s ===")
    logging.info(f"run config: alpha={cfg['alpha']} seed={cfg['seed']} "
                 f"r={cfg['tag_ratio']} mask={cfg['mask_mode']} "
                 f"original ilp_time_limit={cfg.get('ilp_time_limit')}")

    tags, names = rebuild_tags(cfg)
    if tags.shape[0] != assignments.shape[0]:
        raise SystemExit(
            f"Tag/assignment length mismatch ({tags.shape[0]} vs "
            f"{assignments.shape[0]}). The tag matrix was not rebuilt from the "
            f"same split/seed as the run — refusing to report numbers.")
    logging.info(f"Rebuilt tags {tags.shape}, mean {tags.mean():.4f}; "
                 f"assignments {assignments.shape}, "
                 f"{len(np.unique(assignments))} non-empty clusters")

    K = summary["n_clusters"]
    trace, accepted, K_act, lower_bound = scan_beta(
        assignments, tags, K, alpha=cfg["alpha"], time_limit=args.time_limit,
        beta_max=args.beta_max, threads=args.threads, beta_min=args.beta_min)

    out = {
        "run_dir": args.run_dir,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "beta_min": args.beta_min,
        "alpha": cfg["alpha"],
        "seed": cfg["seed"],
        "k_active": K_act,
        "beta_lower_bound_arithmetic": lower_bound,
        "trace": trace,
        "reference": {
            "beta": summary["ilp_last"]["beta"],
            "n_tags": summary["ilp_last"]["n_tags"],
            "avg_tc": summary["description_metrics"]["avg_tc"],
            "avg_itf": summary["description_metrics"]["avg_itf"],
            "original_time_limit": cfg.get("ilp_time_limit"),
        },
    }

    if accepted is not None:
        beta, W_np, active, entry = accepted
        _, dm = description_metrics(assignments, tags, W_np, active,
                                    predicate_names=names)
        out["accepted"] = {"beta": beta, "n_tags": entry["n_tags"],
                           "tags_per_cluster": entry["tags_per_cluster"],
                           "outcome": entry["outcome"],
                           "avg_tc": dm["avg_tc"], "avg_itf": dm["avg_itf"]}
        logging.info(f"ACCEPTED beta={beta} ({entry['outcome']}): "
                     f"tags={entry['n_tags']} TC={dm['avg_tc']} "
                     f"ITF={dm['avg_itf']} nats")
        ref = out["reference"]
        logging.info(f"RUN REPORTED beta={ref['beta']} tags={ref['n_tags']} "
                     f"TC={ref['avg_tc']} ITF={ref['avg_itf']} "
                     f"(limit {ref['original_time_limit']}s)")
    else:
        logging.warning("No verified solution found within the scan range")

    out_dir = os.path.join(args.output_root, "ilp_probe")
    guard_output_root(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    name = (f"{os.path.basename(os.path.normpath(args.run_dir))}"
            f"_t{args.time_limit}_b{args.beta_min}.json")
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(out, f, indent=2)
    logging.info(f"Wrote {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    main()