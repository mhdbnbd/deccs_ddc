"""Algorithm 1 of the DDC paper, with instrumentation.

Paraphrase of the paper's Algorithm 1 (section 3.5):

  1. initialise the network f_theta and the tag space function g (g = identity)
  2. pre-train f_theta on the overall loss of Eq (7)
  3. then repeat until the network and the explanations converge:
       a. build the description ILP of Eq (2-4), searching upward from beta = 0
          in steps of 1 until it is feasible; the solution W* defines g
       b. for each mini-batch: generate the pairwise constraints from Eq (5),
          evaluate L_P (Eq 6) and L_MI (Eq 1), and update f_theta on Eq (7)

Two things the archived attempt got structurally wrong and this file does not:
the ILP mask actually reaches the pairwise loss, and the optimizer is created
once for the whole run.
"""

import csv
import json
import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import clustering_acc
from ddc_v2.model import DDCEncoder
from ddc_v2.objectives import (mi_terms, generate_pairs, pairwise_loss,
                               solve_description_ilp, description_metrics)

HISTORY_FIELDS = [
    "step", "phase", "round", "epoch", "loss_total", "loss_mi", "loss_pairwise",
    "h_y_given_x", "h_y", "active_argmax", "active_mass", "mean_max_prob",
    "n_pairs", "nmi", "acc", "ari", "ilp_beta", "ilp_n_tags", "seconds",
]


class DDCTrainer:
    def __init__(self, features, tags_observed, annotated_mask, true_labels,
                 n_clusters, args, predicate_names=None, device="cuda"):
        self.X = torch.from_numpy(features).float()
        self.T = torch.from_numpy(tags_observed).float()
        self.annotated = torch.from_numpy(annotated_mask.astype(np.bool_))
        self.y = np.asarray(true_labels)
        self.K = n_clusters
        self.args = args
        self.predicate_names = predicate_names
        self.device = torch.device(device)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.model = DDCEncoder(
            in_dim=self.X.shape[1], hidden_dim=args.hidden_dim,
            n_clusters=n_clusters, temperature=args.temperature,
        ).to(self.device)

        # Paper: "the optimizer is Adam with default parameters". Created ONCE.
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.g_mask = torch.ones(self.T.shape[1], device=self.device)
        self.history = []
        self.step = 0
        self.t0 = time.time()
        self.last_ilp = {"beta": None, "n_tags": int(self.T.shape[1])}
        self._beta_hint = 1

    # -- one pass over the data ------------------------------------------------

    def run_epoch(self, phase, rnd, epoch):
        n = self.X.shape[0]
        perm = torch.randperm(n)
        acc = {"total": 0.0, "mi": 0.0, "pw": 0.0, "hyx": 0.0, "hy": 0.0,
               "pairs": 0, "batches": 0}

        for start in range(0, n, self.args.batch_size):
            idx = perm[start:start + self.args.batch_size]
            if idx.numel() < 4:
                continue
            xb = self.X[idx].to(self.device, non_blocking=True)
            tb = self.T[idx].to(self.device, non_blocking=True)
            ann = self.annotated[idx].to(self.device)

            probs = self.model(xb)

            h_yx, h_y = mi_terms(probs)
            loss_mi = h_yx - self.args.mu * h_y

            gt = tb * self.g_mask.unsqueeze(0)          # g(t_i) = t_i * G
            cand = ann if self.args.pairs_from == "annotated" else None
            anchors, partners = generate_pairs(
                probs.detach(), gt, top_l=self.args.top_l,
                gamma=self.args.gamma, candidate_mask=cand,
            )
            loss_pw = pairwise_loss(probs, anchors, partners,
                                    symmetric=self.args.symmetric_kl)

            loss = loss_mi + self.args.lam * loss_pw       # Eq (7)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.grad_clip)
            self.opt.step()
            self.step += 1

            acc["total"] += loss.item()
            acc["mi"] += loss_mi.item()
            acc["pw"] += float(loss_pw.item())
            acc["hyx"] += h_yx.item()
            acc["hy"] += h_y.item()
            acc["pairs"] += int(anchors.numel())
            acc["batches"] += 1

        nb = max(acc["batches"], 1)
        row = self.evaluate()
        row.update({
            "step": self.step, "phase": phase, "round": rnd, "epoch": epoch,
            "loss_total": round(acc["total"] / nb, 6),
            "loss_mi": round(acc["mi"] / nb, 6),
            "loss_pairwise": round(acc["pw"] / nb, 6),
            "h_y_given_x": round(acc["hyx"] / nb, 6),
            "h_y": round(acc["hy"] / nb, 6),
            "n_pairs": acc["pairs"],
            "ilp_beta": self.last_ilp["beta"],
            "ilp_n_tags": self.last_ilp["n_tags"],
            "seconds": round(time.time() - self.t0, 1),
        })
        self.history.append(row)
        logging.info(
            f"[{phase} r{rnd} e{epoch}] L={row['loss_total']:+.4f} "
            f"MI={row['loss_mi']:+.4f} PW={row['loss_pairwise']:.4f} "
            f"H(Y|X)={row['h_y_given_x']:.3f} H(Y)={row['h_y']:.3f} "
            f"active={row['active_argmax']}/{self.K} pairs={row['n_pairs']} "
            f"NMI={row['nmi']:.4f} ACC={row['acc']:.4f}"
        )
        return row

    # -- monitoring ------------------------------------------------------------

    def evaluate(self):
        probs = self.model.predict_probs(self.X, self.device)
        assign = probs.argmax(dim=1).numpy()
        marginal = probs.mean(dim=0)
        return {
            "active_argmax": int(len(np.unique(assign))),
            "active_mass": int((marginal > 0.005).sum().item()),
            "mean_max_prob": round(float(probs.max(dim=1).values.mean()), 4),
            "nmi": round(float(normalized_mutual_info_score(self.y, assign)), 4),
            "acc": round(float(clustering_acc(self.y, assign)), 4),
            "ari": round(float(adjusted_rand_score(self.y, assign)), 4),
        }

    def assignments(self):
        return self.model.predict_probs(self.X, self.device).argmax(dim=1).numpy()

    # -- Algorithm 1 -----------------------------------------------------------

    def fit(self):
        logging.info(f"Pre-training {self.args.pretrain_epochs} epochs "
                     f"(g = identity, Algorithm 1 lines 1-2)")
        for e in range(1, self.args.pretrain_epochs + 1):
            self.run_epoch("pretrain", 0, e)

        best = {"nmi": -1.0}
        for rnd in range(1, self.args.rounds + 1):
            assign = self.assignments()
            g_np, W, info = solve_description_ilp(
                assign, self.T.numpy(), self.K, alpha=self.args.alpha,
                time_limit=self.args.ilp_time_limit,
                beta_start=max(1, self._beta_hint - 1),
            )
            if info["beta"]:
                self._beta_hint = info["beta"]
            self.g_mask = torch.from_numpy(g_np).float().to(self.device)
            self.last_ilp = {"beta": info["beta"], "n_tags": info["n_tags"]}
            self.last_W, self.last_info = W, info

            for e in range(1, self.args.epochs_per_round + 1):
                row = self.run_epoch("round", rnd, e)
                if row["nmi"] > best["nmi"]:
                    best = dict(row)

        self.best = best
        return best

    # -- artefacts -------------------------------------------------------------

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            w.writeheader()
            for row in self.history:
                w.writerow({k: row.get(k) for k in HISTORY_FIELDS})

        assign = self.assignments()
        np.save(os.path.join(output_dir, "assignments.npy"), assign)

        W = getattr(self, "last_W", None)
        info = getattr(self, "last_info", {})
        rows, desc_metrics = description_metrics(
            assign, self.T.numpy(), W, info.get("active_clusters", []),
            predicate_names=self.predicate_names,
        )
        with open(os.path.join(output_dir, "ilp_descriptions.json"), "w") as f:
            json.dump(rows, f, indent=2)

        final = self.evaluate()
        summary = {
            "config": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                           else str(v)) for k, v in vars(self.args).items()},
            "n_samples": int(self.X.shape[0]),
            "n_clusters": self.K,
            "n_attributes": int(self.T.shape[1]),
            "final": final,
            "best_nmi_epoch": getattr(self, "best", {}),
            "description_metrics": desc_metrics,
            "ilp_last": {k: v for k, v in info.items() if k != "active_clusters"},
            "total_seconds": round(time.time() - self.t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        self.plot(output_dir)
        logging.info(f"Artefacts written to {output_dir}/")
        return summary

    def plot(self, output_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib unavailable — skipping curves.png")
            return

        h = self.history
        x = [r["step"] for r in h]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

        ax[0].plot(x, [r["loss_mi"] for r in h], label="L_MI")
        ax[0].plot(x, [r["loss_pairwise"] for r in h], label="L_P")
        ax[0].plot(x, [r["loss_total"] for r in h], label="L total", ls="--")
        ax[0].set_title("loss terms"); ax[0].set_xlabel("step"); ax[0].legend()

        ax[1].plot(x, [r["h_y_given_x"] for r in h], label="H(Y|X)")
        ax[1].plot(x, [r["h_y"] for r in h], label="H(Y)")
        ax[1].axhline(np.log(self.K), color="grey", ls=":", label="log K")
        ax2 = ax[1].twinx()
        ax2.plot(x, [r["active_argmax"] for r in h], color="tab:red",
                 label="active clusters")
        ax2.set_ylabel("active clusters")
        ax[1].set_title("entropies / cluster usage"); ax[1].set_xlabel("step")
        ax[1].legend(loc="upper left")

        ax[2].plot(x, [r["nmi"] for r in h], label="NMI")
        ax[2].plot(x, [r["acc"] for r in h], label="ACC")
        ax[2].plot(x, [r["ari"] for r in h], label="ARI")
        if self.args.baseline_nmi:
            ax[2].axhline(self.args.baseline_nmi, color="k", ls="--",
                          label=f"k-means {self.args.baseline_nmi}")
        ax[2].set_ylim(0, 1); ax[2].set_title("clustering quality")
        ax[2].set_xlabel("step"); ax[2].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "curves.png"), dpi=130)
        plt.close(fig)