"""The three DDC objectives, written directly from the paper's equations.

Equation numbers refer to Zhang & Davidson, "Deep Descriptive Clustering",
IJCAI 2021 (proceedings PDF, pp. 3342-3348).
"""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Eq (1): clustering objective
# ---------------------------------------------------------------------------

def mi_terms(probs):
    """
    L_MI = -I(X;Y) = H(Y|X) - H(Y)
         = 1/N sum_i h(p(y_i|x_i)) - h(1/N sum_i p(y_i|x_i))          (Eq 1)

    h is the entropy function; natural log throughout. The marginal is the mean
    over the current mini-batch, which is what "solve an approximated version ...
    in each mini-batch" implies for the batched form.

    Returns (h_y_given_x, h_y) as scalars with gradient.
    """
    p = probs.clamp(min=EPS)
    h_y_given_x = -(p * p.log()).sum(dim=1).mean()
    marginal = p.mean(dim=0).clamp(min=EPS)
    h_y = -(marginal * marginal.log()).sum()
    return h_y_given_x, h_y


# ---------------------------------------------------------------------------
# Eq (5): self-generated together-constraints
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_pairs(probs, masked_tags, top_l=1, gamma=100.0, candidate_mask=None):
    """
    J_i = min_j  gamma * |g(t_i) - g(t_j)|  -  |f_theta(x_i) - f_theta(x_j)|   (Eq 5)

    Section 3.4: for each instance, take the top l partners minimising J as
    self-generated together-constraints; since evaluating Eq (5) over the whole
    training set is impractical, N is replaced by the batch size and an
    approximate version is solved inside each mini-batch.

    So: within the batch, pick the l partners with the smallest J — close in the
    masked tag space, and currently far apart in the clustering output. Those are
    exactly the inconsistencies the pairwise loss is meant to repair.

    JUDGMENT CALLS (paper is silent):
      - |.| is taken as the L2 norm.
      - f_theta(x) is taken as the network's cluster-probability output, which is
        what "f_theta(x_i) != f_theta(x_j)" refers to in section 3.4.
      - candidate_mask restricts *partners* to annotated instances. With
        instance-level r-masking every unannotated instance carries the same
        imputed tag vector, so leaving them in makes them each other's nearest
        neighbours and the constraints become uninformative.

    Returns (anchor_idx, partner_idx) as LongTensors into the batch.
    """
    n = probs.shape[0]
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=probs.device)
        return empty, empty

    tag_dist = torch.cdist(masked_tags, masked_tags, p=2)
    clus_dist = torch.cdist(probs, probs, p=2)
    j_score = gamma * tag_dist - clus_dist
    j_score.fill_diagonal_(float("inf"))

    if candidate_mask is not None:
        blocked = ~candidate_mask
        if blocked.all():
            empty = torch.empty(0, dtype=torch.long, device=probs.device)
            return empty, empty
        j_score[:, blocked] = float("inf")

    l = int(min(top_l, n - 1))
    _, partners = torch.topk(j_score, l, dim=1, largest=False)

    anchors = torch.arange(n, device=probs.device).unsqueeze(1).expand(-1, l)
    anchors = anchors.reshape(-1)
    partners = partners.reshape(-1)

    finite = torch.isfinite(j_score[anchors, partners])
    if candidate_mask is not None:
        # only annotated anchors carry real tag information
        finite = finite & candidate_mask[anchors]
    return anchors[finite], partners[finite]


# ---------------------------------------------------------------------------
# Eq (6): pairwise loss
# ---------------------------------------------------------------------------

def pairwise_loss(probs, anchors, partners, symmetric=False):
    """
    L_P = 1/(N l) sum_i sum_j KL( p(y_i|x_i), p(y_j|x_j) )                (Eq 6)

    Directed KL(p_i || p_j) by default, matching the equation as written.
    symmetric=True averages both directions (ablation only).
    """
    if anchors.numel() == 0:
        return probs.sum() * 0.0

    p = probs.clamp(min=EPS)
    p_i, p_j = p[anchors], p[partners]
    kl = F.kl_div(p_j.log(), p_i, reduction="batchmean")
    if symmetric:
        kl = 0.5 * (kl + F.kl_div(p_i.log(), p_j, reduction="batchmean"))
    return kl


# ---------------------------------------------------------------------------
# Eq (2-4): cluster-level explanation ILP, and the mask function g
# ---------------------------------------------------------------------------

def solve_description_ilp(assignments, tags, n_clusters, alpha=8,
                          beta_max=None, time_limit=30, beta_start=1):
    """
    min sum_ij W_ij                                                      (Eq 2)
    s.t. sum_j W_ij Q_ij >= alpha    for every cluster i                 (Eq 3)
         sum_i W_ij Q_ij <= beta     for every tag j                     (Eq 4)
         W_ij in {0,1}
    with Q_ij = (1/|C_i|) sum_{t_k in C_i} t_kj  and mean imputation for
    missing tags.

    Algorithm 1 lines 4-8: beta starts at 0 and is increased by a fixed step of 1
    until the ILP admits a feasible W*, which then defines the tag space function
    g. Beta = 0 is trivially infeasible for alpha > 0, so the search starts at 1.

    Returns (g_mask (M,) float32 numpy, W (K_act, M) numpy, info dict).
    g_mask[j] = 1 iff tag j appears in the solution — this is the diagonal of G
    in section 3.3.
    """
    import pulp

    M = tags.shape[1]
    active = [k for k in range(n_clusters) if (assignments == k).sum() > 0]
    K_act = len(active)
    if K_act < 2:
        logging.warning("[ILP] fewer than 2 non-empty clusters — g left as identity")
        return np.ones(M, dtype=np.float32), None, {"beta": None, "n_tags": M,
                                                    "k_active": K_act, "status": "skipped"}

    Q = np.zeros((K_act, M), dtype=np.float64)
    for row, k in enumerate(active):
        Q[row] = tags[assignments == k].mean(axis=0)

    beta_max = beta_max or K_act
    beta_start = max(1, min(int(beta_start), beta_max))
    trace = []
    for beta in range(beta_start, beta_max + 1):
        prob = pulp.LpProblem("DDC_description", pulp.LpMinimize)
        W = {(i, j): pulp.LpVariable(f"W_{i}_{j}", cat="Binary")
             for i in range(K_act) for j in range(M)}
        prob += pulp.lpSum(W[i, j] for i in range(K_act) for j in range(M))
        for i in range(K_act):
            prob += pulp.lpSum(W[i, j] * Q[i, j] for j in range(M)) >= alpha
        for j in range(M):
            prob += pulp.lpSum(W[i, j] * Q[i, j] for i in range(K_act)) <= beta
        t_solve = time.time()
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        t_solve = time.time() - t_solve

        # CBC termination is NOT readable from prob.status alone: PuLP maps
        # "Stopped on time - objective value X" onto LpStatusOptimal and records
        # the difference only in sol_status, so a time-limited incumbent would
        # otherwise be logged as a proven optimum. Checked against pulp 3.3.2,
        # pulp/apis/coin_api.py (cbcStatus / cbcSolStatus).
        sol_code = getattr(prob, "sol_status", None)
        if prob.status == pulp.LpStatusOptimal:
            outcome = ("proven_optimal" if sol_code == pulp.LpSolutionOptimal
                       else "time_limit_incumbent")
        elif prob.status == pulp.LpStatusInfeasible:
            outcome = "proven_infeasible"
        else:
            outcome = "time_limit_no_solution"   # unproven; treated as infeasible

        entry = {"beta": beta, "outcome": outcome,
                 "lp_status": pulp.LpStatus.get(prob.status, "unknown"),
                 "sol_status": pulp.LpSolution.get(sol_code, "unknown"),
                 "seconds": round(t_solve, 1), "time_limit": time_limit,
                 "objective": (round(float(pulp.value(prob.objective)), 2)
                               if prob.status == pulp.LpStatusOptimal else None)}
        trace.append(entry)

        if prob.status != pulp.LpStatusOptimal:
            logging.info(f"[ILP] beta={beta} rejected: {outcome} "
                         f"({t_solve:.1f}s of {time_limit}s)")
            continue

        W_np = np.zeros((K_act, M), dtype=np.float32)
        for i in range(K_act):
            for j in range(M):
                W_np[i, j] = W[i, j].varValue or 0.0
        W_np = (W_np > 0.5).astype(np.float32)

        cover = (W_np * Q).sum(axis=1)
        load = (W_np * Q).sum(axis=0)
        ok = bool((cover >= alpha - 1e-6).all() and (load <= beta + 1e-6).all())
        entry["constraints_verified"] = ok
        if not ok:
            logging.warning(f"[ILP] beta={beta} returned W violating Eq (3)/(4) "
                            f"(min coverage {cover.min():.3f} vs alpha={alpha}, "
                            f"max load {load.max():.3f} vs beta={beta}) - rejected")
            continue

        info = {"beta": beta, "n_tags": int((W_np.sum(axis=0) > 0.5).sum()),
                "k_active": K_act,
                "tags_per_cluster": float(W_np.sum(axis=1).mean()),
                "status": outcome, "optimality_proven": outcome == "proven_optimal",
                "time_limit": time_limit, "solve_seconds": round(t_solve, 1),
                "trace": trace, "active_clusters": active}
        g_mask = (W_np.sum(axis=0) > 0.5).astype(np.float32)
        logging.info(f"[ILP] beta={beta} [{outcome}, {t_solve:.1f}s], {K_act} active "
                     f"clusters, {info['n_tags']}/{M} tags kept, "
                     f"{info['tags_per_cluster']:.1f} tags/cluster")
        return g_mask, W_np, info

    if beta_start > 1:
        # warm start overshot; fall back to the full search
        g_mask, W_np, info = solve_description_ilp(
            assignments, tags, n_clusters, alpha=alpha, beta_max=beta_max,
            time_limit=time_limit, beta_start=1)
        info["trace"] = trace + info.get("trace", [])
        return g_mask, W_np, info
    logging.warning(f"[ILP] no accepted solution for beta <= {beta_max}; "
                    f"g left as identity")
    return np.ones(M, dtype=np.float32), None, {"beta": None, "n_tags": M,
                                                "k_active": K_act,
                                                "status": "infeasible",
                                                "optimality_proven": False,
                                                "time_limit": time_limit,
                                                "trace": trace}


def description_metrics(assignments, tags, W, active_clusters, predicate_names=None):
    """
    TC(C_i) = 1/|D_i| sum_{d in D_i} |{(x,t) in C_i : d in t}| / |C_i|     (Eq 8)
    ITF(C_i) = 1/|D_i| sum_{d in D_i} log( K / sum_j |d in D_j| )          (Eq 9)

    NOTE: the paper gives the ITF range as [0, log K] without fixing a base; this
    project uses the natural log everywhere, so ITF is reported in nats. Do not
    compare these numbers directly against the paper's Table 2 figures, which are
    base-2 (their max value 2.32 = log2 5).
    """
    if W is None:
        return [], {"avg_tc": 0.0, "avg_itf": 0.0}

    M = tags.shape[1]
    names = predicate_names or [f"attr_{j}" for j in range(M)]
    K_act = W.shape[0]
    tag_use = (W > 0.5).sum(axis=0).astype(np.float64)

    rows = []
    for row, k in enumerate(active_clusters):
        selected = np.where(W[row] > 0.5)[0]
        members = assignments == k
        if len(selected) == 0 or members.sum() == 0:
            rows.append({"cluster_id": int(k), "n_members": int(members.sum()),
                         "tags": [], "tc": 0.0, "itf": 0.0})
            continue
        tc = float(np.mean([tags[members, j].mean() for j in selected]))
        itf = float(np.mean([np.log(K_act / max(tag_use[j], 1.0)) for j in selected]))
        rows.append({
            "cluster_id": int(k), "n_members": int(members.sum()),
            "tags": [names[j] for j in selected],
            "tag_indices": selected.tolist(),
            "tc": round(tc, 4), "itf": round(itf, 4),
        })

    avg_tc = float(np.mean([r["tc"] for r in rows])) if rows else 0.0
    avg_itf = float(np.mean([r["itf"] for r in rows])) if rows else 0.0
    return rows, {"avg_tc": round(avg_tc, 4), "avg_itf": round(avg_itf, 4)}