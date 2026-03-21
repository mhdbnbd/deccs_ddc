"""
Training functions for DECCS-DDC integration.

Three training modes:
  - train_autoencoder: reconstruction-only (baseline AE)
  - train_constrained_autoencoder: reconstruction + tag BCE (baseline CAE)
  - train_ddc: MI clustering + pairwise constraints + optional consensus (DDCNet)

The train_ddc function implements DDC's actual training objectives:
  L = L_MI + lambda_pairwise * L_pairwise [+ lambda_consensus * L_consensus]

  L_MI: Mutual information maximization between inputs and cluster assignments.
        = -H(Y) + H(Y|X)  where we minimize H(Y|X) and maximize H(Y)
        Concretely: encourage confident per-sample assignments (low entropy)
        while keeping cluster usage uniform (high marginal entropy).

  L_pairwise: For each instance, find tag-similar instances and force their
              cluster assignments to agree (KL divergence).
              This is DDC's key mechanism for connecting tags to clustering.

  L_consensus: DECCS extension — pull embeddings toward ensemble agreement.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


def make_grad_scaler(device):
    try:
        return GradScaler(device_type=device, enabled=(device == "cuda"))
    except TypeError:
        return GradScaler(enabled=(device == "cuda"))


def _prepare_model_for_cuda(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.to(memory_format=torch.channels_last)
    return model


# =============================================================================
# Baseline AE training (reconstruction only)
# =============================================================================

def train_autoencoder(dataloader, model, use_gpu, num_epochs=1, **kwargs):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        model = _prepare_model_for_cuda(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device.type)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            if batch is None:
                continue
            images = batch[0] if len(batch) >= 2 else batch
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        logging.info(f"[AE] Epoch {epoch + 1}/{num_epochs} | Loss={avg_loss:.6f}")

    return avg_loss


# =============================================================================
# Baseline constrained AE training (reconstruction + tag BCE)
# =============================================================================

def train_constrained_autoencoder(
        dataloader, model, use_gpu, num_epochs=1,
        tag_tuner=0.5, consensus_matrix=None, lambda_consensus=0.0, **kwargs
):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    if device.type == 'cuda':
        model = _prepare_model_for_cuda(model)

    recon_loss_fn = nn.MSELoss()
    tag_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = make_grad_scaler(device.type)

    has_decoder = None
    model.train()
    running_loss = running_recon = running_tag = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue

        images, symbolic_tags, idx = batch
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        symbolic_tags = symbolic_tags.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            first_output, tag_logits = model(images)

            if has_decoder is None:
                has_decoder = (first_output.shape == images.shape)
                logging.info(f"Model {'has' if has_decoder else 'lacks'} decoder")

            tag_loss = tag_loss_fn(tag_logits, symbolic_tags)
            recon_loss = recon_loss_fn(first_output, images) if has_decoder else torch.tensor(0.0, device=device)

            if consensus_matrix is not None and lambda_consensus > 0:
                with torch.no_grad():
                    z = model.get_embeddings(images)
                idx_np = idx.cpu().numpy() if isinstance(idx, torch.Tensor) else idx
                consensus_sub = consensus_matrix[np.ix_(idx_np, idx_np)]
                if hasattr(consensus_sub, 'toarray'):
                    consensus_sub = consensus_sub.toarray()
                from utils import consensus_consistency_loss
                cons_loss = consensus_consistency_loss(z, consensus_sub)
                total_loss = recon_loss + tag_tuner * tag_loss + lambda_consensus * cons_loss
            else:
                total_loss = recon_loss + tag_tuner * tag_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.detach().item()
        running_recon += recon_loss.detach().item()
        running_tag += tag_loss.detach().item()
        n_batches += 1

    if n_batches == 0:
        return {"total": 0.0, "recon": 0.0, "tag": 0.0}

    avg = lambda x: x / n_batches
    logging.info(f"Recon={avg(running_recon):.4f}, Tag={avg(running_tag):.4f}, Total={avg(running_loss):.4f}")
    return {"total": avg(running_loss), "recon": avg(running_recon), "tag": avg(running_tag)}


# =============================================================================
# DDC training: MI loss + pairwise constraints + optional DECCS consensus
#
# This is the KEY function that was missing from the previous implementation.
# It implements DDC's actual training objectives rather than reconstruction + BCE.
# =============================================================================

def mi_loss(cluster_probs):
    """MI clustering loss: L_MI = H(Y|X) - H(Y). Minimizing maximizes I(X;Y)."""
    eps = 1e-8
    probs = cluster_probs.clamp(min=eps)
    h_y_given_x = -(probs * torch.log(probs)).sum(dim=1).mean()
    marginal = probs.mean(dim=0)
    h_y = -(marginal * torch.log(marginal)).sum()
    return h_y_given_x, h_y


def solve_ilp_for_mask(cluster_probs, tags, n_clusters, alpha=8):
    """
    DDC's ILP (Eq. 2-4): find concise, orthogonal cluster descriptions.

    Uses PuLP solver for exact ILP solution (DDC paper's approach).
    Solves for K×M binary matrix W where W_ij=1 iff cluster i described by tag j.

    Objective: minimize total tags used (Eq. 2)
    Constraint 1 (coverage): sum_j(W_ij * Q_ij) >= alpha per cluster (Eq. 3)
    Constraint 2 (orthogonality): sum_i(W_ij * Q_ij) <= beta per tag (Eq. 4)
    Beta is searched from 1 upward until feasible (DDC Algorithm 1).
    """
    import pulp

    probs_np = cluster_probs.detach().cpu().numpy()
    tags_np = tags.detach().cpu().numpy()
    N, M = tags_np.shape
    K = n_clusters

    # Hard cluster assignments → Q matrix
    assignments = probs_np.argmax(axis=1)
    Q = np.zeros((K, M))
    for k in range(K):
        mask_k = (assignments == k)
        if mask_k.sum() > 0:
            Q[k] = tags_np[mask_k].mean(axis=0)

    active = [k for k in range(K) if (assignments == k).sum() > 0]
    K_act = len(active)
    if K_act < 2:
        return torch.ones(M, device=cluster_probs.device), None

    Q_act = Q[active]

    # Search for smallest feasible beta (DDC Algorithm 1)
    for beta in range(1, K_act + 1):
        prob = pulp.LpProblem("DDC_ILP", pulp.LpMinimize)

        # Binary variables W_ij
        W_vars = {}
        for i in range(K_act):
            for j in range(M):
                W_vars[i, j] = pulp.LpVariable(f"W_{i}_{j}", cat='Binary')

        # Objective: minimize total tags
        prob += pulp.lpSum(W_vars[i, j] for i in range(K_act) for j in range(M))

        # Coverage: sum_j(W_ij * Q_ij) >= alpha for each active cluster
        for i in range(K_act):
            prob += pulp.lpSum(W_vars[i, j] * Q_act[i, j] for j in range(M)) >= alpha

        # Orthogonality: sum_i(W_ij * Q_ij) <= beta for each tag
        for j in range(M):
            prob += pulp.lpSum(W_vars[i, j] * Q_act[i, j] for i in range(K_act)) <= beta

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=10))

        if prob.status == 1:  # Optimal
            W_np = np.zeros((K_act, M))
            for i in range(K_act):
                for j in range(M):
                    W_np[i, j] = W_vars[i, j].varValue or 0

            mask_np = (W_np.sum(axis=0) > 0.5).astype(np.float32)
            n_tags = int(mask_np.sum())
            logging.info(f"[ILP] beta={beta}, {K_act} active clusters, "
                         f"{n_tags}/{M} tags selected, "
                         f"tags/cluster={W_np.sum(1).mean():.1f}")
            return (torch.tensor(mask_np, device=cluster_probs.device),
                    torch.tensor(W_np, dtype=torch.float32))

    # Fallback to greedy if ILP fails
    logging.warning(f"[ILP] No feasible solution — falling back to greedy")
    W_np = np.zeros((K_act, M))
    tag_usage = np.zeros(M)
    for i in range(K_act):
        scores = Q_act[i] / (1.0 + tag_usage)
        top = np.argsort(scores)[-alpha:]
        W_np[i, top] = 1
        tag_usage[top] += 1
    mask_np = (W_np.sum(axis=0) > 0.5).astype(np.float32)
    n_tags = int(mask_np.sum())
    logging.info(f"[ILP-fallback] {K_act} clusters, {n_tags}/{M} tags selected")
    return (torch.tensor(mask_np, device=cluster_probs.device),
            torch.tensor(W_np, dtype=torch.float32))


def pairwise_constraint_loss(cluster_probs, tags, mask=None, top_l=5, gamma=100.0):
    """
    DDC's pairwise constraint loss (DDC Eq. 7-8).

    For each instance i, find top_l instances that are SIMILAR in tag space
    but DIFFERENT in cluster assignment space. These are "errors" — instances
    that should be together based on their attributes but aren't.

    The loss minimizes KL(p_i || p_j) for each such pair, forcing the network
    to assign tag-similar instances to the same cluster.

    Args:
        cluster_probs: (N, K) softmax cluster probabilities
        tags: (N, T) semantic attribute vectors
        top_l: number of constraint pairs per instance
        gamma: weight for tag distance (DDC uses gamma=100)
    """
    N = cluster_probs.size(0)
    if N < 2:
        return torch.tensor(0.0, device=cluster_probs.device)

    # Apply ILP mask to tags (DDC's g function: g(t_i) = t_i * G)
    if mask is not None:
        masked_tags = tags * mask.unsqueeze(0)  # (N, M) * (1, M) → (N, M)
    else:
        masked_tags = tags

    probs = cluster_probs.clamp(min=1e-8)

    # Pair selection using MASKED tags (no gradients needed)
    with torch.no_grad():
        tag_dist = torch.cdist(masked_tags, masked_tags, p=2)
        cluster_dist = torch.cdist(probs.detach(), probs.detach(), p=2)
        selection_score = gamma * tag_dist - cluster_dist
        selection_score.fill_diagonal_(float('inf'))
        actual_l = min(top_l, N - 1)
        _, pair_indices = selection_score.topk(actual_l, dim=1, largest=False)

    # KL divergence between selected pairs (WITH gradients)
    log_probs = torch.log(probs)

    loss = 0.0
    for l in range(actual_l):
        partner_probs = probs[pair_indices[:, l]]
        partner_log = torch.log(partner_probs)
        kl_fwd = F.kl_div(log_probs, partner_probs, reduction='batchmean', log_target=False)
        kl_rev = F.kl_div(partner_log, probs, reduction='batchmean', log_target=False)
        loss += (kl_fwd + kl_rev) / 2.0

    return loss / actual_l


def train_ddc(
        dataloader,
        model,
        use_gpu,
        num_epochs=1,
        lambda_pairwise=1.0,
        lambda_tag=0.0,
        tag_ratio=0.5,
        consensus_matrix=None,
        lambda_consensus=0.0,
        tag_tuner=None,
        _cached_features=None,
        **kwargs
):
    """
    Train DDCNet matching DDC paper (Zhang & Davidson, IJCAI 2021) exactly.

    Loss = lambda * L_P + L_MI  (DDC Eq. 7)

    Key DDC design choices:
      - Mini-batch training with per-batch pairwise constraints
      - Per-instance tag masking (r=0.5): randomly zero 50% of each instance's
        tags. CRITICAL because AwA2 tags are per-class — without masking,
        same-class pairs have identical tags (dist=0) and pairwise selection
        ignores tag information entirely.
      - l=1: one constraint pair per instance
      - lambda=1, gamma=100, Adam lr=1e-3
    """
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Pre-extract frozen ResNet features (once)
    if _cached_features is None:
        logging.info("Pre-extracting ResNet-101 backbone features...")
        all_features, all_tags, all_indices = model.extract_backbone_features(dataloader, device)
        logging.info(f"Extracted features: {all_features.shape}")
    else:
        all_features, all_tags, all_indices = _cached_features

    N = all_features.shape[0]
    batch_size = min(256, N)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    model.train()

    running = {"total": 0.0, "mi": 0.0, "pairwise": 0.0}
    n_batches = 0

    # Mini-batch training over pre-extracted features
    perm = torch.randperm(N)
    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        feat_batch = all_features[idx].to(device)
        tags_batch = all_tags[idx].to(device)

        # Per-instance random tag masking (DDC's r=0.5)
        mask = (torch.rand_like(tags_batch) >= tag_ratio).float()
        masked_tags = tags_batch * mask

        optimizer.zero_grad()
        out = model.forward_from_features(feat_batch)
        cluster_probs = out['cluster_probs']

        # L_MI (DDC Eq. 1)
        h_y_given_x, h_y = mi_loss(cluster_probs)
        loss_mi = h_y_given_x - h_y

        # L_P with masked tags, l=1 (DDC Eq. 5-6)
        loss_pw = pairwise_constraint_loss(cluster_probs, masked_tags, top_l=1)

        # L = lambda * L_P + L_MI (DDC Eq. 7)
        total_loss = loss_mi + lambda_pairwise * loss_pw

        # Optional DECCS consensus
        if consensus_matrix is not None and lambda_consensus > 0:
            batch_indices = all_indices[idx]
            idx_np = batch_indices.cpu().numpy() if isinstance(batch_indices, torch.Tensor) else batch_indices
            consensus_sub = consensus_matrix[np.ix_(idx_np, idx_np)]
            if hasattr(consensus_sub, 'toarray'):
                consensus_sub = consensus_sub.toarray()
            from utils import consensus_consistency_loss
            total_loss = total_loss + lambda_consensus * consensus_consistency_loss(
                out['embeddings'], consensus_sub)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running["total"] += total_loss.detach().item()
        running["mi"] += loss_mi.detach().item()
        running["pairwise"] += loss_pw.detach().item()
        n_batches += 1

    if n_batches == 0:
        return {"total": 0.0, "recon": 0.0, "tag": 0.0,
                "_cached_features": (all_features, all_tags, all_indices)}

    avg = {k: v / n_batches for k, v in running.items()}

    # Monitoring on full dataset
    with torch.no_grad():
        out_full = model.forward_from_features(all_features.to(device))
        p = out_full['cluster_probs']
        n_used = (p.mean(0) > 0.005).sum().item()
        max_p = p.max(1).values.mean().item()
        hyx = -(p.clamp(1e-8) * p.clamp(1e-8).log()).sum(1).mean()
        hy = -(p.mean(0).clamp(1e-8) * p.mean(0).clamp(1e-8).log()).sum()
        logging.info(f"MI={avg['mi']:.4f} (H(Y|X)={hyx:.4f}, H(Y)={hy:.4f}), "
                     f"PW={avg['pairwise']:.4f}, Total={avg['total']:.4f}, "
                     f"Active={n_used}/{p.size(1)}, MaxProb={max_p:.4f}")

    return {
        "total": avg["total"], "recon": avg["mi"], "tag": avg["pairwise"],
        "_cached_features": (all_features, all_tags, all_indices),
    }


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_autoencoder(dataloader, model, use_gpu):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            loss = criterion(model(images), images)
            total_loss += loss.item()
    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss