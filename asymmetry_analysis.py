"""
asymmetry_analysis.py
=====================
Rank-2 analysis: Confusion Matrix Directionality (Amnesia vs Palimpsest)

For each temperature condition, reruns the Exp 4B cooling classifier on
cycle-400 glasses only, saves per-sample predictions, and computes:

    α(T) = [P(fast→slow) - P(slow→fast)] / [P(fast→slow) + P(slow→fast)]

α > 0  → Palimpsest / Mechanical Annealing  (fast glass looks like slow after fatigue)
α ≈ 0  → Amnesia / Mechanical Disordering   (both classes drift to a new disordered state)
α < 0  → Reverse Annealing                  (unexpected, would indicate over-disordering of slow glasses)

Physical prediction from PEL framework:
  Low T  (deep glass):  α ≈ 0   — elastic regime, symmetric disordering
  High T (near Tg):     α > 0   — thermally activated annealing, fast→slow convergence

Outputs:
  - asymmetry_results/asymmetry_results.pkl   (all raw data)
  - asymmetry_results/fig_asymmetry.png       (publication figure)
  - asymmetry_results/asymmetry_summary.csv   (table for paper)

Usage:
    python asymmetry_analysis.py --data_dir ./multihistory_data --out_dir ./asymmetry_results
    python asymmetry_analysis.py --data_dir /kaggle/input/.../multihistory_data --out_dir ./asymmetry_results
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURE_CONDITIONS = [
    ("amp_8_t_35",  0.35, 8,  "T=0.35\n(0.78Tg)"),
    ("amp_8_t_38",  0.38, 8,  "T=0.38\n(0.84Tg)"),
    ("amp_8_t_40",  0.40, 8,  "T=0.40\n(0.89Tg)"),
    ("amp_8_t_42",  0.42, 8,  "T=0.42\n(0.93Tg)"),
    ("amp_8_t_44",  0.44, 8,  "T=0.44\n(0.98Tg)"),
    ("amp_8_t_46",  0.46, 8,  "T=0.46\n(1.02Tg)"),
]

STRAIN_CONDITIONS = [
    ("amp_4_t_42",  0.42, 4,  "δ=4%"),
    ("amp_6_t_42",  0.42, 6,  "δ=6%"),
    ("amp_8_t_42",  0.42, 8,  "δ=8%\n(base)"),
    ("amp_12_t_42", 0.42, 12, "δ=12%"),
]

# Ground-truth interference drops from paper (for cross-validation panel)
KNOWN_DROPS = {
    "amp_8_t_35":  14, "amp_8_t_38":  27, "amp_8_t_40":  20,
    "amp_8_t_42":  25, "amp_8_t_44":  19, "amp_8_t_46":  31,
    "amp_4_t_42":  24, "amp_6_t_42":  20, "amp_12_t_42": 24,
}

# Training hyperparameters (matched to paper)
SEED       = 42
N_FOLDS    = 5
LR         = 3e-4
WD         = 1e-4
MAX_EPOCHS = 150
PATIENCE   = 25
BATCH_SIZE = 32
HIDDEN     = 64
HEADS      = 4
N_LAYERS   = 2
DROPOUT    = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# GNN MODEL  (single-task cooling classifier)
# ─────────────────────────────────────────────────────────────────────────────

class GATv2CoolingClassifier(nn.Module):
    """
    Matches SharedEncoder + single-task head in train.py exactly.
    HIDDEN=64, HEADS=4, concat=True -> GAT output = 64*4 = 256
    post-MLP: 256 -> 128 (LN+ReLU+Drop) -> 64 (ReLU) -> 1
    """
    def __init__(self, in_dim=8, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.enc  = nn.Linear(in_dim, hidden)
        self.gat1 = GATv2Conv(hidden, hidden,
                               heads=heads, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(hidden * heads, hidden,
                               heads=heads, edge_dim=1, concat=True)
        pool_dim  = hidden * heads   # 256
        self.post = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = F.relu(self.enc(x))
        h = F.relu(self.gat1(h, ei, ea))
        h = F.relu(self.gat2(h, ei, ea))
        g = global_mean_pool(h, batch)
        z = self.post(g)
        return self.head(z).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (ported from train.py — must be identical)
# ─────────────────────────────────────────────────────────────────────────────

BOX_L    = float((256 / 1.2) ** (1.0 / 3.0))   # ≈ 5.975 σ
RC_GRAPH = 1.5                                   # first-shell cutoff


def extract_features(positions: np.ndarray,
                     box: float = BOX_L,
                     rc: float  = RC_GRAPH) -> tuple:
    """
    8D local bond-length features with instance normalisation.
    Identical to the extract_features() in train.py.

    Returns (node_feat (N,8), edge_index (2,E), edge_attr (E,1))
    """
    N   = len(positions)
    pos = np.asarray(positions, dtype=np.float32)

    # Pairwise distances — minimum-image PBC
    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box * np.round(dr / box)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))   # (N, N)
    nbr  = (dist > 1e-6) & (dist < rc)

    feats = np.zeros((N, 8), dtype=np.float32)
    coord = nbr.sum(axis=1).astype(np.float32)
    d_max = coord.max() if coord.max() > 0 else 1.0

    for i in range(N):
        nd = dist[i][nbr[i]]
        if len(nd) == 0:
            feats[i] = [rc, 0., rc, rc, 0., 0., rc, rc]
            continue
        mu  = nd.mean()
        sig = nd.std() if len(nd) > 1 else 0.0
        sk  = float(np.mean((nd - mu)**3) / (sig**3 + 1e-8)) if sig > 1e-8 else 0.0
        feats[i, 0] = mu
        feats[i, 1] = sig
        feats[i, 2] = nd.min()
        feats[i, 3] = nd.max()
        feats[i, 4] = coord[i] / d_max
        feats[i, 5] = float(np.clip(sk, -5., 5.))
        feats[i, 6] = float(np.percentile(nd, 25))
        feats[i, 7] = float(np.percentile(nd, 75))

    # Instance normalisation (per-graph z-score)
    mu_g  = feats.mean(axis=0, keepdims=True)
    sig_g = feats.std(axis=0,  keepdims=True) + 1e-8
    feats = (feats - mu_g) / sig_g

    # Edges
    rows, cols = np.where(nbr)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)

    return feats, edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_cycle400_graphs(pkl_path: str):
    """
    Load only cycle-400 glasses from a raw positions pickle.
    Features are computed on-the-fly from positions using extract_features().
    Returns (list[Data], np.ndarray of labels).
    """
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    graphs, labels = [], []
    for s in samples:
        if s["label_fatigue"] != 2:   # 2 = cycle 400
            continue

        positions = np.array(s["positions"], dtype=np.float32)   # (N, 3)

        bond_feats, edges, edge_attr = extract_features(positions)

        if not (np.isfinite(bond_feats).all() and np.isfinite(edge_attr).all()):
            continue   # skip NaN samples (should not occur)

        data = Data(
            x          = torch.tensor(bond_feats),
            edge_index = torch.tensor(edges),
            edge_attr  = torch.tensor(edge_attr),
            y          = torch.tensor(float(s["label_cooling"])),
        )
        graphs.append(data)
        labels.append(int(s["label_cooling"]))

    return graphs, np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = F.binary_cross_entropy_with_logits(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / sum(b.num_graphs for b in loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs  = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(probs)
        labels_all.append(batch.y.cpu().numpy())
    probs_all  = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    preds = (probs_all >= 0.5).astype(int)
    acc   = (preds == labels_all).mean() * 100
    auc   = roc_auc_score(labels_all, probs_all)
    return acc, auc, preds, probs_all, labels_all


def run_single_fold(train_graphs, val_graphs, seed=0):
    """Train one fold. Returns (acc, auc, preds, probs, y_true)."""
    torch.manual_seed(seed)
    model = GATv2CoolingClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(model, train_loader, optimizer)
        scheduler.step()

        # Compute val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch)
                loss   = F.binary_cross_entropy_with_logits(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_graphs)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, auc, preds, probs, y_true = evaluate(model, val_loader)
    return acc, auc, preds, probs, y_true


# ─────────────────────────────────────────────────────────────────────────────
# ASYMMETRY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_asymmetry(y_true, y_pred):
    """
    α = [P(fast→slow) - P(slow→fast)] / [P(fast→slow) + P(slow→fast)]

    Labels: 0=fast, 1=slow
    Errors: fast→slow = predicted 1 when true 0
            slow→fast = predicted 0 when true 1
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm[i, j] = predicted j when true i
    # cm[0, 1] = fast predicted as slow   (fast→slow error)
    # cm[1, 0] = slow predicted as fast   (slow→fast error)
    n_fast_total = cm[0].sum()
    n_slow_total = cm[1].sum()

    if n_fast_total == 0 or n_slow_total == 0:
        return np.nan, np.nan, np.nan

    p_fast_to_slow = cm[0, 1] / n_fast_total   # FPR
    p_slow_to_fast = cm[1, 0] / n_slow_total   # FNR

    denom = p_fast_to_slow + p_slow_to_fast
    if denom < 1e-9:
        return 0.0, p_fast_to_slow, p_slow_to_fast

    alpha = (p_fast_to_slow - p_slow_to_fast) / denom
    return alpha, p_fast_to_slow, p_slow_to_fast


def bootstrap_alpha(y_true, y_pred, n_boot=2000, seed=0):
    """Bootstrap CI for α."""
    rng = np.random.default_rng(seed)
    alphas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        a, _, _ = compute_asymmetry(yt, yp)
        if not np.isnan(a):
            alphas.append(a)
    alphas = np.array(alphas)
    return np.percentile(alphas, [2.5, 97.5])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS LOOP
# ─────────────────────────────────────────────────────────────────────────────

def analyse_condition(condition_key, data_dir):
    """Run 5-fold CV on cycle-400 glasses, return full asymmetry results."""
    pkl_path = os.path.join(data_dir, f"multihistory_glasses_{condition_key}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Data file not found: {pkl_path}")

    print(f"  Loading {pkl_path} ...", flush=True)
    graphs, labels = load_cycle400_graphs(pkl_path)
    print(f"    {len(graphs)} cycle-400 glasses  "
          f"(fast={sum(labels==0)}, slow={sum(labels==1)})", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_preds  = np.zeros(len(graphs), dtype=int)
    all_probs  = np.zeros(len(graphs))
    all_true   = labels.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
        train_g = [graphs[i] for i in train_idx]
        val_g   = [graphs[i] for i in val_idx]

        acc, auc, preds, probs, y_true = run_single_fold(
            train_g, val_g, seed=SEED + fold_idx
        )
        all_preds[val_idx] = preds
        all_probs[val_idx] = probs

        alpha, p_fs, p_sf = compute_asymmetry(y_true, preds)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])

        fold_results.append({
            "fold":     fold_idx + 1,
            "acc":      acc,
            "auc":      auc,
            "alpha":    alpha,
            "p_fast_to_slow": p_fs,
            "p_slow_to_fast": p_sf,
            "cm":       cm,
        })
        print(f"    fold {fold_idx+1}: acc={acc:.1f}%  AUC={auc:.4f}  "
              f"α={alpha:+.3f}  "
              f"(fast→slow={p_fs:.3f}, slow→fast={p_sf:.3f})", flush=True)

    # Aggregate
    mean_acc   = np.mean([r["acc"]   for r in fold_results])
    std_acc    = np.std( [r["acc"]   for r in fold_results])
    mean_alpha = np.mean([r["alpha"] for r in fold_results
                          if not np.isnan(r["alpha"])])
    std_alpha  = np.std( [r["alpha"] for r in fold_results
                          if not np.isnan(r["alpha"])])

    # Bootstrap CI on pooled predictions
    boot_ci = bootstrap_alpha(all_true, all_preds)

    # Pooled confusion matrix
    pooled_cm = sum(r["cm"] for r in fold_results)
    pooled_alpha, pooled_pfs, pooled_psf = compute_asymmetry(all_true, all_preds)

    print(f"    → acc={mean_acc:.1f}±{std_acc:.1f}%  "
          f"α={mean_alpha:+.3f}±{std_alpha:.3f}  "
          f"(pooled α={pooled_alpha:+.3f}, 95% CI [{boot_ci[0]:+.3f}, {boot_ci[1]:+.3f}])",
          flush=True)

    return {
        "condition":       condition_key,
        "fold_results":    fold_results,
        "mean_acc":        mean_acc,
        "std_acc":         std_acc,
        "mean_alpha":      mean_alpha,
        "std_alpha":       std_alpha,
        "pooled_alpha":    pooled_alpha,
        "pooled_p_fs":     pooled_pfs,
        "pooled_p_sf":     pooled_psf,
        "boot_ci":         boot_ci,
        "pooled_cm":       pooled_cm,
        "all_preds":       all_preds,
        "all_true":        all_true,
        "all_probs":       all_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_asymmetry_figure(temp_results, strain_results, out_path):
    """
    Publication figure: 4 panels
    A  α(T) for temperature sweep — the main mechanistic result
    B  α(δ) for strain sweep     — confirms strain doesn't control mechanism
    C  Pooled confusion matrices at T=0.35 and T=0.46
    D  α vs interference drop scatter — decorrelation test
    """
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")

    T_VALS   = [0.35, 0.38, 0.40, 0.42, 0.44, 0.46]
    T_TG     = [f"{t/0.45:.2f}" for t in T_VALS]
    D_VALS   = [4, 6, 8, 12]

    # Unpack
    temp_alpha  = [r["pooled_alpha"] for r in temp_results]
    temp_ci_lo  = [r["boot_ci"][0]   for r in temp_results]
    temp_ci_hi  = [r["boot_ci"][1]   for r in temp_results]
    temp_acc    = [r["mean_acc"]      for r in temp_results]

    strain_alpha = [r["pooled_alpha"] for r in strain_results]
    strain_ci_lo = [r["boot_ci"][0]   for r in strain_results]
    strain_ci_hi = [r["boot_ci"][1]   for r in strain_results]

    # ── Panel A: α(T) ─────────────────────────────────────────────────────
    ax_a = fig.add_subplot(2, 3, 1)
    x = np.arange(len(T_VALS))
    err_lo = [a - lo for a, lo in zip(temp_alpha, temp_ci_lo)]
    err_hi = [hi - a for a, hi in zip(temp_alpha, temp_ci_hi)]

    cols = ["#e74c3c" if a > 0.05 else "#3498db" if a < -0.05 else "#95a5a6"
            for a in temp_alpha]

    ax_a.bar(x, temp_alpha, color=cols, alpha=0.85, zorder=3, width=0.6)
    ax_a.errorbar(x, temp_alpha,
                  yerr=[err_lo, err_hi],
                  fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4)
    ax_a.axhline(0,   color="black",  linewidth=1.2, linestyle="-")
    ax_a.axhline(0.1, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.axhline(-0.1,color="#3498db", linewidth=0.8, linestyle="--", alpha=0.5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"T={t}\n({tg}Tg)" for t, tg in zip(T_VALS, T_TG)],
                          fontsize=7.5)
    ax_a.set_ylabel("Asymmetry ratio  α", fontsize=10)
    ax_a.set_title("(A) Temperature sweep  (δ=8%)", fontweight="bold", fontsize=10)
    ax_a.set_ylim(-0.7, 1.0)

    # Region labels
    ax_a.text(0.02, 0.97, "Palimpsest\n(fast→slow)\nα > 0",
              transform=ax_a.transAxes, fontsize=7, va="top",
              color="#e74c3c", alpha=0.7)
    ax_a.text(0.02, 0.35, "Amnesia\n(symmetric)\nα ≈ 0",
              transform=ax_a.transAxes, fontsize=7, va="top",
              color="#95a5a6", alpha=0.7)
    ax_a.text(0.02, 0.18, "Reverse\nα < 0",
              transform=ax_a.transAxes, fontsize=7, va="top",
              color="#3498db", alpha=0.7)
    ax_a.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel B: α(δ) ─────────────────────────────────────────────────────
    ax_b = fig.add_subplot(2, 3, 2)
    xs = np.arange(len(D_VALS))
    err_lo_s = [a - lo for a, lo in zip(strain_alpha, strain_ci_lo)]
    err_hi_s = [hi - a for a, hi in zip(strain_alpha, strain_ci_hi)]

    cols_s = ["#e74c3c" if a > 0.05 else "#3498db" if a < -0.05 else "#95a5a6"
              for a in strain_alpha]
    ax_b.bar(xs, strain_alpha, color=cols_s, alpha=0.85, zorder=3, width=0.6)
    ax_b.errorbar(xs, strain_alpha,
                  yerr=[err_lo_s, err_hi_s],
                  fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4)
    ax_b.axhline(0,   color="black",  linewidth=1.2)
    ax_b.axhline(0.1, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.axhline(-0.1,color="#3498db", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"δ={d}%\n(T=0.42)" for d in D_VALS], fontsize=8)
    ax_b.set_ylabel("Asymmetry ratio  α", fontsize=10)
    ax_b.set_title("(B) Strain sweep  (T=0.42)", fontweight="bold", fontsize=10)
    ax_b.set_ylim(-0.7, 1.0)
    ax_b.grid(axis="y", alpha=0.3, zorder=0)
    mean_strain_a = np.mean(strain_alpha)
    ax_b.axhline(mean_strain_a, color="gray", linestyle=":", linewidth=1.2)
    ax_b.text(xs[-1] + 0.1, mean_strain_a + 0.02,
              f"mean={mean_strain_a:+.3f}", fontsize=7.5, color="gray", va="bottom")

    # ── Panel C: Confusion matrices at T=0.35 and T=0.46 ─────────────────
    for panel_col, (res, label) in enumerate(
            zip([temp_results[0], temp_results[5]],
                ["T=0.35  (0.78Tg)\nDeep glass", "T=0.46  (1.02Tg)\nAbove Tg"])):

        ax_c = fig.add_subplot(2, 3, 4 + panel_col)
        cm = res["pooled_cm"].astype(float)
        # Normalise by row (true class)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        im = ax_c.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax_c.set_xticks([0, 1]);  ax_c.set_xticklabels(["Pred: fast", "Pred: slow"])
        ax_c.set_yticks([0, 1]);  ax_c.set_yticklabels(["True: fast", "True: slow"])
        ax_c.set_title(f"({'CD'[panel_col]}) {label}\nα={res['pooled_alpha']:+.3f}",
                        fontweight="bold", fontsize=9)

        for i in range(2):
            for j in range(2):
                ax_c.text(j, i,
                          f"{cm_norm[i,j]:.2f}\n(n={int(cm[i,j])})",
                          ha="center", va="center", fontsize=9,
                          color="white" if cm_norm[i,j] > 0.6 else "black")

        # Annotate dominant error
        a = res["pooled_alpha"]
        if a > 0.1:
            ax_c.text(0.5, -0.22,
                      f"Palimpsest: fast glasses mis-classified\nas slow after fatigue",
                      transform=ax_c.transAxes, ha="center", fontsize=7.5,
                      color="#e74c3c", style="italic")
        elif abs(a) <= 0.1:
            ax_c.text(0.5, -0.22,
                      "Amnesia: symmetric misclassification",
                      transform=ax_c.transAxes, ha="center", fontsize=7.5,
                      color="#7f8c8d", style="italic")

    # ── Panel D: α vs Interference Drop ───────────────────────────────────
    ax_d = fig.add_subplot(2, 3, 3)

    all_alpha = [r["pooled_alpha"] for r in temp_results + strain_results]
    all_drop  = [KNOWN_DROPS[r["condition"]] for r in temp_results + strain_results]
    all_T     = [r["condition"].split("_t_")[1] for r in temp_results + strain_results]

    # Colour: temp sweep = red, strain sweep = blue
    colors_d = (["#e74c3c"] * len(temp_results) +
                ["#2980b9"] * len(strain_results))

    ax_d.scatter(all_alpha, all_drop, c=colors_d, s=80, zorder=5, alpha=0.9)
    for a, d, c, lbl in zip(all_alpha, all_drop, temp_results + strain_results, all_T):
        ax_d.annotate(
            f"T={lbl}" if lbl.isdigit() or "." in lbl else lbl,
            (a, d), textcoords="offset points", xytext=(5, 3), fontsize=7
        )

    # Spearman correlation
    rho, pval = stats.spearmanr(all_alpha, all_drop)
    ax_d.set_xlabel("Asymmetry ratio α  (pooled)", fontsize=10)
    ax_d.set_ylabel("Interference drop  (pp)", fontsize=10)
    ax_d.set_title(f"(D) α  vs  interference drop\nSpearman ρ={rho:.3f}  p={pval:.3f}",
                    fontweight="bold", fontsize=10)
    ax_d.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_d.grid(alpha=0.3)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Temperature sweep")
    blue_patch = mpatches.Patch(color="#2980b9", label="Strain sweep")
    ax_d.legend(handles=[red_patch, blue_patch], fontsize=8)

    # ── Panel E: P(fast→slow) and P(slow→fast) side by side ──────────────
    ax_e = fig.add_subplot(2, 3, 6)

    pfs_vals = [r["pooled_p_fs"] for r in temp_results]
    psf_vals = [r["pooled_p_sf"] for r in temp_results]
    xpos = np.arange(len(T_VALS))

    ax_e.plot(xpos, pfs_vals, "o-", color="#e74c3c", label="P(fast→slow)", linewidth=2)
    ax_e.plot(xpos, psf_vals, "s-", color="#2980b9", label="P(slow→fast)", linewidth=2)
    ax_e.fill_between(xpos, pfs_vals, psf_vals,
                       where=[f > s for f, s in zip(pfs_vals, psf_vals)],
                       alpha=0.15, color="#e74c3c", label="Palimpsest region")
    ax_e.fill_between(xpos, pfs_vals, psf_vals,
                       where=[f <= s for f, s in zip(pfs_vals, psf_vals)],
                       alpha=0.15, color="#2980b9", label="Reverse region")
    ax_e.set_xticks(xpos)
    ax_e.set_xticklabels([f"T={t}" for t in T_VALS], fontsize=8)
    ax_e.set_ylabel("Misclassification probability", fontsize=10)
    ax_e.set_title("(E) Error rates by direction\n(temperature sweep)", fontweight="bold", fontsize=10)
    ax_e.legend(fontsize=7.5)
    ax_e.grid(alpha=0.3)
    ax_e.set_ylim(0, 1)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(temp_results, strain_results, out_path):
    rows = []
    for res in temp_results + strain_results:
        key = res["condition"]
        t_str = key.split("_t_")[1]
        d_str = key.split("_t_")[0].replace("amp_", "")
        rows.append({
            "Condition":        f"T={float(t_str)/100:.2f}, δ={d_str}%"
                                if False else key,
            "T (LJ)":           float(t_str) / 100,
            "δ (%)":            int(d_str),
            "acc_4B (%)":       f"{res['mean_acc']:.1f}±{res['std_acc']:.1f}",
            "α (pooled)":       f"{res['pooled_alpha']:+.3f}",
            "95% CI":           f"[{res['boot_ci'][0]:+.3f}, {res['boot_ci'][1]:+.3f}]",
            "P(fast→slow)":     f"{res['pooled_p_fs']:.3f}",
            "P(slow→fast)":     f"{res['pooled_p_sf']:.3f}",
            "Mechanism":        ("Palimpsest" if res["pooled_alpha"] > 0.1
                                 else "Amnesia" if abs(res["pooled_alpha"]) <= 0.1
                                 else "Reverse"),
            "Interference (pp)": KNOWN_DROPS.get(res["condition"], "—"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  CSV saved → {out_path}")
    print("\n" + df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./multihistory_data",
                        help="Directory containing multihistory_glasses_*.pkl files")
    parser.add_argument("--out_dir",  default="./asymmetry_results",
                        help="Output directory for figures and CSVs")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip conditions whose pkl result already exists in out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(SEED)

    print(f"\n{'='*60}")
    print("  CONFUSION MATRIX ASYMMETRY ANALYSIS")
    print(f"  Device: {DEVICE}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Output: {args.out_dir}")
    print(f"{'='*60}\n")

    all_conditions = TEMPERATURE_CONDITIONS + [c for c in STRAIN_CONDITIONS
                                                if c[0] != "amp_8_t_42"]
    # baseline (amp_8_t_42) is already in temp conditions

    results_cache_path = os.path.join(args.out_dir, "asymmetry_results.pkl")

    # Check for cached results
    if args.skip_existing and os.path.exists(results_cache_path):
        print(f"  Loading cached results from {results_cache_path}")
        with open(results_cache_path, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    # Run analysis for each condition
    for cond_key, T, delta, label in TEMPERATURE_CONDITIONS + STRAIN_CONDITIONS:
        if cond_key in all_results and args.skip_existing:
            print(f"  [SKIP] {cond_key} (already computed)")
            continue

        print(f"\n{'─'*60}")
        print(f"  Condition: {cond_key}  (T={T}, δ={delta}%)")
        print(f"{'─'*60}")

        try:
            result = analyse_condition(cond_key, args.data_dir)
            all_results[cond_key] = result
        except FileNotFoundError as e:
            print(f"  [WARNING] {e}")
            print(f"  Skipping {cond_key}")

    # Save raw results
    with open(results_cache_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n  Raw results saved → {results_cache_path}")

    # Separate into temp and strain sweeps for plotting
    temp_results   = [all_results[c[0]] for c in TEMPERATURE_CONDITIONS
                      if c[0] in all_results]
    strain_results = [all_results[c[0]] for c in STRAIN_CONDITIONS
                      if c[0] in all_results]

    # Generate outputs
    print(f"\n{'='*60}")
    print("  GENERATING OUTPUTS")
    print(f"{'='*60}")

    fig_path = os.path.join(args.out_dir, "fig_asymmetry.png")
    make_asymmetry_figure(temp_results, strain_results, fig_path)

    csv_path = os.path.join(args.out_dir, "asymmetry_summary.csv")
    df = make_summary_table(temp_results, strain_results, csv_path)

    # Print key interpretation
    print(f"\n{'='*60}")
    print("  KEY RESULTS")
    print(f"{'='*60}")
    print(f"\n  Temperature sweep α values:")
    for res, (cond_key, T, *_) in zip(temp_results, TEMPERATURE_CONDITIONS):
        ci = res["boot_ci"]
        mech = ("PALIMPSEST" if res["pooled_alpha"] > 0.1
                else "AMNESIA" if abs(res["pooled_alpha"]) <= 0.1
                else "REVERSE")
        print(f"    T={T}:  α={res['pooled_alpha']:+.3f}  "
              f"95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]  → {mech}")

    print(f"\n  Strain sweep α values (T=0.42):")
    for res, (cond_key, T, delta, *_) in zip(strain_results, STRAIN_CONDITIONS):
        ci = res["boot_ci"]
        print(f"    δ={delta}%: α={res['pooled_alpha']:+.3f}  "
              f"95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]")

    # Spearman between α and interference drop
    all_res_list = temp_results + strain_results
    all_alpha = [r["pooled_alpha"] for r in all_res_list]
    all_drop  = [KNOWN_DROPS[r["condition"]] for r in all_res_list]
    rho, pval = stats.spearmanr(all_alpha, all_drop)
    print(f"\n  Spearman(α, interference_drop): ρ={rho:.3f}  p={pval:.3f}")
    print(f"  {'Correlated' if pval < 0.05 else 'NOT correlated'} — "
          f"{'α and interference drop co-vary' if pval < 0.05 else 'α and interference drop are INDEPENDENT'}")

    print(f"\n  Done. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()