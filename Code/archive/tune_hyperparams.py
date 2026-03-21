import json
import itertools
import logging
import subprocess
from pathlib import Path

lambda_grid = [0.05, 0.1, 0.2]
tag_tuner_grid = [0.5, 1.0, 1.5]
epochs = 10
mode = "deccs"
use_sample = True

logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_experiment(lc, tt):
    """Run DECCS with given hyperparams and return final metrics."""
    out_json = f"results_tune_lc{lc}_tt{tt}.json"
    cmd = [
        "python3", "main_experiments.py",
        "--mode", mode,
        "--epochs", str(epochs),
        "--use_sample",
        "--lambda_consensus", str(lc),
        "--tag_tuner", str(tt),
        "--output_json", out_json
    ]
    logging.info(f" Running lambda_consensus={lc}, tag_tuner={tt}")
    subprocess.run(cmd, check=True)
    if Path(out_json).exists():
        with open(out_json) as f:
            res = json.load(f)
        return res["results"]["metrics"]
    return None

results = []
for lc, tt in itertools.product(lambda_grid, tag_tuner_grid):
    metrics = run_experiment(lc, tt)
    if metrics:
        results.append({"lambda_consensus": lc, "tag_tuner": tt, **metrics})

output_json = "hyperparam_results.json"
with open(output_json, "w") as f:
    json.dump(results, f, indent=4)
logging.info(f"Results saved to {output_json}")

best = max(results, key=lambda x: x["nmi"])
logging.info("\n=== Best Configuration ===")
logging.info(f"lambda_consensus={best['lambda_consensus']}, tag_tuner={best['tag_tuner']} → NMI={best['nmi']:.4f}, Sil={best['silhouette']:.3f}")

# Append summary to log file
with open("tuning_summary.log", "a") as log:
    for r in results:
        log.write(f"lambda_consensus={r['lambda_consensus']}, tag_tuner={r['tag_tuner']}, NMI={r['nmi']:.4f}, Sil={r['silhouette']:.3f}\n")
    log.write(f"Best → lambda_consensus={best['lambda_consensus']}, tag_tuner={best['tag_tuner']}, NMI={best['nmi']:.4f}\n\n")
