#!/usr/bin/env python3
"""
==========================================================================
ARRHENIUS FORGETTING CURVE  —  Rank-1 Experiment
Paper: "Geometric Orthogonality of Competing Structural Memories in Glass"
==========================================================================

WHAT THIS SCRIPT DOES
─────────────────────
For each of 6 cycling temperatures T ∈ {0.35, 0.38, 0.40, 0.42, 0.44, 0.46}:

  Step 1 — Train a GATv2 binary cooling classifier (fast vs slow)
           separately on glasses at EACH of the 9 cycle snapshots:
           c ∈ {0, 50, 100, 150, 200, 250, 300, 350, 400}.
           This gives acc(c, T) — the GNN's ability to detect thermal
           history as a function of how many fatigue cycles have elapsed.

  Step 2 — Fit the forgetting curve acc(c, T) to a stretched exponential:
               acc(c, T) = A∞(T) + [A0 − A∞(T)] · exp[−(c/τ(T))^β]
           Extracts the timescale τ(T) at which thermal memory is lost.

  Step 3 — Fit τ(T) to the Arrhenius equation:
               log τ(T) = log τ₀ + Eₐ / T
           Extracts the activation energy Eₐ in LJ reduced units.

  Step 4 — Compare Eₐ to known cooperative rearrangement barriers in
           LJ glass from mode-coupling theory (MCT) and inherent-structure
           calculations. If Eₐ matches, the GNN probe is measuring the
           same physical process as classical glass relaxation theory.

EXPECTED RUNTIME (RTX 4060)
────────────────────────────
  9 cycle points × 5 folds × 6 temperatures = 270 training runs
  Each run: ~200 samples, max 150 epochs
  ≈ 3–5 hours total on RTX 4060
  ≈ 4–6 hours total on Kaggle T4

  Use --skip_existing to resume from cached forgetting_curves.pkl
  if interrupted.

OUTPUTS
────────
  arrhenius_results/
    forgetting_curves.pkl     — raw acc(c,T) arrays with fold-level variance
    arrhenius_results.pkl     — τ(T), β(T), A∞(T), Eₐ, R² of Arrhenius fit
    fig_forgetting_curves.png — 6-panel forgetting curve figure
    fig_arrhenius.png         — Arrhenius plot: log τ vs 1/T
    arrhenius_summary.csv     — table for paper

USAGE
──────
  # Local RTX 4060
  python arrhenius_forgetting.py --data_dir D:\\final_exp\\paper_v3\\multihistory_data --out_dir .\\arrhenius_results

  # Kaggle
  python arrhenius_forgetting.py --data_dir /kaggle/input/.../multihistory_data --out_dir ./arrhenius_results

  # Resume if interrupted
  python arrhenius_forgetting.py --data_dir ... --out_dir ... --skip_existing
==========================================================================
"""

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURES = [
    # (filename_suffix, T_value, T/Tg_label)
    # suffix is appended to: multihistory_glasses_{suffix}.pkl
    #
    # T=0.35, 0.38, 0.40, 0.46 — generated with original datagen script
    ("amp_8_t_35_cycle_9",   0.35, "0.78Tg"),
    ("amp_8_t_38_cycle_9",   0.38, "0.84Tg"),
    ("amp_8_t_40_cycle_9",   0.40, "0.89Tg"),
    # T=0.42, 0.44 — generated with dual-GPU Kaggle script (no underscore after amp8)
    # T=0.44 has a typo in filename: "cylce" instead of "cycle" — kept as-is
    ("amp8_t_42_cycle_9",    0.42, "0.93Tg"),
    ("amp8_t_44_cylce_9",    0.44, "0.98Tg"),   # note: cylce typo in filename
    ("amp_8_t_46_cycle_9",   0.46, "1.02Tg"),
]

SAVE_AT    = [0, 50, 100, 150, 200, 250, 300, 350, 400]
MAX_CYCLES = 400
TG         = 0.45

# Training (matched to train.py)
SEED       = 42
N_FOLDS    = 5
LR         = 3e-4
WD         = 1e-4
MAX_EPOCHS = 150
PATIENCE   = 25
BATCH_SIZE = 32
HIDDEN     = 64
HEADS      = 4

# MCT cooperative rearrangement barrier for LJ glass at rho=1.2
# Literature: ~3-6 epsilon (LJ reduced units)
# Key references: Kob & Andersen (1995), Sciortino et al. (1999),
# inherent-structure barrier distributions in monodisperse LJ glass
MCT_EA_LO = 3.0
MCT_EA_HI = 6.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────

BOX_L    = float((256 / 1.2) ** (1.0 / 3.0))
RC_GRAPH = 1.5


def extract_features(positions, box=BOX_L, rc=RC_GRAPH):
    N   = len(positions)
    pos = np.asarray(positions, dtype=np.float32)
    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box * np.round(dr / box)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))
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

    mu_g  = feats.mean(axis=0, keepdims=True)
    sig_g = feats.std(axis=0,  keepdims=True) + 1e-8
    feats = (feats - mu_g) / sig_g

    rows, cols = np.where(nbr)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)
    return feats, edge_index, edge_attr


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_graphs_at_cycle(pkl_path, cycle):
    """Load samples where fatigue_cycles == cycle. Returns (graphs, labels)."""
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    graphs, labels = [], []
    for s in samples:
        if s["fatigue_cycles"] != cycle:
            continue
        nf, ei, ea = extract_features(np.array(s["positions"], dtype=np.float32))
        if not (np.isfinite(nf).all() and np.isfinite(ea).all()):
            continue
        graphs.append(Data(
            x          = torch.tensor(nf),
            edge_index = torch.tensor(ei),
            edge_attr  = torch.tensor(ea),
            y          = torch.tensor(float(s["label_cooling"])),
        ))
        labels.append(int(s["label_cooling"]))

    return graphs, np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (matches train.py SharedEncoder + single-task head exactly)
# ─────────────────────────────────────────────────────────────────────────────

class GATv2Classifier(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.enc  = nn.Linear(in_dim, HIDDEN)
        self.gat1 = GATv2Conv(HIDDEN, HIDDEN, heads=HEADS,
                               edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(HIDDEN * HEADS, HIDDEN, heads=HEADS,
                               edge_dim=1, concat=True)
        self.post = nn.Sequential(
            nn.Linear(HIDDEN * HEADS, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = F.relu(self.enc(x))
        h = F.relu(self.gat1(h, ei, ea))
        h = F.relu(self.gat2(h, ei, ea))
        return self.head(self.post(global_mean_pool(h, batch))).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, scaler):
    model.train()
    for b in loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss = F.binary_cross_entropy_with_logits(model(b), b.y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    for b in loader:
        b = b.to(DEVICE)
        probs_all.append(torch.sigmoid(model(b)).cpu().numpy())
        labels_all.append(b.y.cpu().numpy())
    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    acc    = ((probs >= 0.5).astype(int) == labels).mean() * 100
    loss   = float(F.binary_cross_entropy_with_logits(
        torch.tensor(probs), torch.tensor(labels.astype(np.float32))).item())
    return loss, acc


def run_cv_at_cycle(graphs, labels, cycle):
    """5-fold CV on one cycle snapshot. Returns list of 5 fold accuracies."""
    if len(graphs) < 10:
        print(f"    WARNING: only {len(graphs)} samples at cycle {cycle}")
        return [50.0] * N_FOLDS

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    accs   = []

    for fi, (tr_idx, va_idx) in enumerate(skf.split(graphs, labels)):
        torch.manual_seed(SEED + fi)
        model = GATv2Classifier().to(DEVICE)
        opt   = Adam(model.parameters(), lr=LR, weight_decay=WD)
        sch   = CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

        tr_ld = DataLoader([graphs[i] for i in tr_idx],
                           batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS)
        va_ld = DataLoader([graphs[i] for i in va_idx],
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS)

        best_loss, best_acc, pat = float("inf"), 50.0, 0
        for ep in range(MAX_EPOCHS):
            train_epoch(model, tr_ld, opt, scaler)
            sch.step()
            val_loss, val_acc = evaluate(model, va_ld)
            if val_loss < best_loss - 1e-4:
                best_loss, best_acc, pat = val_loss, val_acc, 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break
        accs.append(best_acc)

    print(f"    cycle {cycle:3d}: acc={np.mean(accs):.1f}±{np.std(accs):.1f}%  "
          f"(n={len(graphs)})", flush=True)
    return accs


# ─────────────────────────────────────────────────────────────────────────────
# FORGETTING CURVE FITTING
# ─────────────────────────────────────────────────────────────────────────────

def stretched_exp(c, A_inf, A0, tau, beta):
    """acc(c) = A_inf + (A0 - A_inf) * exp(-(c/tau)^beta)"""
    return A_inf + (A0 - A_inf) * np.exp(-np.power(np.maximum(c / tau, 1e-9), beta))


def fit_forgetting_curve(cycles, mean_accs, std_accs):
    c = np.array(cycles, dtype=float)
    y = np.array(mean_accs, dtype=float)
    w = 1.0 / (np.array(std_accs, dtype=float) + 1.0)

    try:
        popt, pcov = curve_fit(
            stretched_exp, c, y,
            p0=[y[-1], y[0], 150.0, 0.8],
            bounds=([0, 50, 1, 0.1], [100, 100, 2000, 2.0]),
            sigma=w, maxfev=20000
        )
        A_inf, A0, tau, beta = popt
        perr = np.sqrt(np.diag(pcov))
        y_pred = stretched_exp(c, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = max(0.0, 1 - ss_res / max(ss_tot, 1e-10))
        return dict(A_inf=float(A_inf), A_inf_err=float(perr[0]),
                    A0=float(A0),       A0_err=float(perr[1]),
                    tau=float(tau),     tau_err=float(perr[2]),
                    beta=float(beta),   beta_err=float(perr[3]),
                    r2=float(r2),       converged=True)
    except Exception as e:
        print(f"    WARNING: curve_fit failed ({e}). Using linear interpolation.")
        # Fallback: estimate tau as cycle where acc drops 50% of total drop
        A0_est = y[0]; Ainf_est = y[-1]
        half   = (A0_est + Ainf_est) / 2.0
        # interpolate
        try:
            tau_est = float(np.interp(half, y[::-1], c[::-1]))
        except Exception:
            tau_est = float(c[len(c)//2])
        return dict(A_inf=float(Ainf_est), A_inf_err=float(np.std(y[-2:])),
                    A0=float(A0_est),     A0_err=0.0,
                    tau=tau_est,          tau_err=float(tau_est * 0.5),
                    beta=1.0,             beta_err=0.0,
                    r2=0.0,               converged=False)


# ─────────────────────────────────────────────────────────────────────────────
# ARRHENIUS FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit_arrhenius(temps, taus, tau_errs):
    """log(tau) = log(tau0) + Ea/T. Returns Ea in LJ epsilon units."""
    T    = np.array(temps)
    tau  = np.array(taus)
    terr = np.array(tau_errs)

    # Use only well-converged points (relative error < 50%)
    mask = (terr / np.maximum(tau, 1.0) < 0.5) & (tau > 0)
    if mask.sum() < 3:
        print("  WARNING: fewer than 3 converged tau values — using all points")
        mask = tau > 0

    T_u   = T[mask]
    tau_u = tau[mask]

    slope, intercept, r, p, se = linregress(1.0 / T_u, np.log(tau_u))

    return dict(
        Ea=float(slope),       Ea_err=float(se * 1.96),
        tau0=float(np.exp(intercept)),
        r2=float(r**2),        p=float(p),
        slope=float(slope),    intercept=float(intercept),
        temps_used=T_u.tolist(), taus_used=tau_u.tolist(),
        n_points=int(mask.sum())
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_forgetting_curves(all_results, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    cmap  = plt.cm.coolwarm
    T_all = [r["T"] for r in all_results]
    T_min, T_max = min(T_all), max(T_all)

    c_dense = np.linspace(0, MAX_CYCLES, 400)

    for idx, (res, ax) in enumerate(zip(all_results, axes.flatten())):
        T   = res["T"]
        col = cmap((T - T_min) / max(T_max - T_min, 1e-6))
        fit = res["fit"]

        cycles    = np.array(SAVE_AT, dtype=float)
        mean_accs = np.array(res["mean_accs"])
        std_accs  = np.array(res["std_accs"])

        ax.errorbar(cycles, mean_accs, yerr=std_accs,
                    fmt="o", color=col, capsize=4, linewidth=1.5,
                    markersize=7, zorder=5, label="GNN acc (5-fold CV)")

        if fit["converged"]:
            ax.plot(c_dense, stretched_exp(c_dense, fit["A_inf"],
                    fit["A0"], fit["tau"], fit["beta"]),
                    "-", color=col, linewidth=2.2, alpha=0.85,
                    label=f"Fit  $R^2={fit['r2']:.3f}$")

        ax.axvline(fit["tau"], color=col, ls="--", alpha=0.45, lw=1.2)
        ax.text(0.97, 0.96,
                f"τ = {fit['tau']:.0f}±{fit['tau_err']:.0f}\n"
                f"β = {fit['beta']:.2f}  A∞ = {fit['A_inf']:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=col, alpha=0.85))

        ax.axhline(50, color="gray", ls=":", lw=0.8)
        ax.set_title(f"T = {T}  ({res['T_Tg']})", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Fatigue cycles", fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel("Cooling-rate accuracy (%)", fontsize=9)
        ax.set_xlim(-10, 420)
        ax.set_ylim(35, 105)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5)

    plt.suptitle(
        "GNN Forgetting Curves: Thermal Memory Decay vs Fatigue Cycles\n"
        r"Fit: $A_\infty + (A_0 - A_\infty)\exp[-(c/\tau)^\beta]$",
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Forgetting curve figure → {out_path}")


def plot_arrhenius(all_results, arr, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Arrhenius ─────────────────────────────────────────────────────────
    ax = axes[0]
    T_all    = [r["T"]              for r in all_results]
    tau_all  = [r["fit"]["tau"]     for r in all_results]
    terr_all = [r["fit"]["tau_err"] for r in all_results]
    conv_all = [r["fit"]["converged"] for r in all_results]

    invT     = np.array([1/T for T in T_all])
    log_tau  = np.log(tau_all)

    for i in range(len(all_results)):
        rel_err = terr_all[i] / max(tau_all[i], 1.0)
        ax.errorbar(invT[i], log_tau[i], yerr=rel_err,
                    fmt="o" if conv_all[i] else "s",
                    color="#e74c3c", capsize=4, markersize=9,
                    markerfacecolor="#e74c3c" if conv_all[i] else "white",
                    markeredgecolor="#e74c3c", lw=1.5, zorder=5)
        ax.annotate(f"T={T_all[i]}", (invT[i], log_tau[i]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)

    # Fit line
    iT_range = np.linspace(min(invT)*0.95, max(invT)*1.05, 100)
    ax.plot(iT_range, arr["slope"]*iT_range + arr["intercept"],
            "-", color="#2c3e50", lw=2,
            label=f"Arrhenius  $R^2={arr['r2']:.3f}$")

    Ea = arr["Ea"]; Ea_err = arr["Ea_err"]
    within_mct = MCT_EA_LO <= Ea <= MCT_EA_HI
    box_col = "#27ae60" if within_mct else "#e67e22"
    ax.text(0.04, 0.97,
            f"$E_a = {Ea:.2f} \\pm {Ea_err:.2f}\\,\\varepsilon$\n"
            f"MCT range: [{MCT_EA_LO:.0f}, {MCT_EA_HI:.0f}] ε\n"
            f"{'✓ Matches MCT barrier' if within_mct else '⚠ Outside MCT range'}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc=box_col, alpha=0.15,
                      ec=box_col))
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlabel("1/T  (LJ units)", fontsize=11)
    ax.set_ylabel("log τ  (log cycles)", fontsize=11)
    ax.set_title("(A) Arrhenius Plot", fontweight="bold")

    # ── τ(T) and β(T) ─────────────────────────────────────────────────────
    ax2 = axes[1]; ax2b = ax2.twinx()
    beta_all = [r["fit"]["beta"] for r in all_results]

    ax2.errorbar(T_all, tau_all, yerr=terr_all,
                 fmt="o-", color="#e74c3c", capsize=4, lw=2, ms=8,
                 label="τ (decay timescale)")
    ax2b.plot(T_all, beta_all, "s--", color="#2980b9", lw=1.5, ms=7,
              label="β (stretching exponent)")

    ax2.axvline(TG, color="black", ls="--", lw=1.2, alpha=0.6)
    ax2.text(TG + 0.002, max(tau_all)*0.05, "$T_g$", fontsize=10)

    ax2.set_xlabel("Cycling temperature T (LJ)", fontsize=11)
    ax2.set_ylabel("τ (cycles)", fontsize=11, color="#e74c3c")
    ax2b.set_ylabel("β", fontsize=11, color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2b.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_title("(B) Timescale τ(T) and β(T)", fontweight="bold")
    ax2.grid(alpha=0.3)

    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1+l2, lab1+lab2, fontsize=9)

    plt.suptitle(
        f"Arrhenius Analysis: $E_a = {Ea:.2f} \\pm {Ea_err:.2f}\\,\\varepsilon$  "
        f"(LJ reduced units)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Arrhenius figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="./multihistory_data")
    parser.add_argument("--out_dir",       default="./arrhenius_results")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Load cached forgetting_curves.pkl — skip retraining")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "forgetting_curves.pkl")

    print(f"\n{'='*60}")
    print(f"  ARRHENIUS FORGETTING CURVE ANALYSIS")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Data   : {args.data_dir}")
    print(f"  Output : {args.out_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: compute or load forgetting curves ─────────────────────────
    if args.skip_existing and os.path.exists(cache_path):
        print(f"  Loading cached curves from {cache_path}")
        with open(cache_path, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = []
        for cond_key, T_val, T_Tg in TEMPERATURES:
            pkl_path = os.path.join(args.data_dir,
                                    f"multihistory_glasses_{cond_key}.pkl")
            if not os.path.exists(pkl_path):
                print(f"  [SKIP] {pkl_path} not found — "
                      f"check filename matches datagen output")
                continue

            print(f"\n{'─'*60}")
            print(f"  T = {T_val}  ({T_Tg})")
            print(f"{'─'*60}")

            mean_accs, std_accs, fold_accs_all = [], [], []
            for cycle in SAVE_AT:
                graphs, labels = load_graphs_at_cycle(pkl_path, cycle)
                fold_accs = run_cv_at_cycle(graphs, labels, cycle)
                fold_accs_all.append(fold_accs)
                mean_accs.append(float(np.mean(fold_accs)))
                std_accs.append(float(np.std(fold_accs)))

            fit = fit_forgetting_curve(SAVE_AT, mean_accs, std_accs)
            print(f"\n  → τ={fit['tau']:.1f}±{fit['tau_err']:.1f}  "
                  f"β={fit['beta']:.3f}  A∞={fit['A_inf']:.1f}%  "
                  f"R²={fit['r2']:.4f}  converged={fit['converged']}")

            all_results.append({
                "cond_key": cond_key, "T": T_val, "T_Tg": T_Tg,
                "cycles": SAVE_AT, "mean_accs": mean_accs,
                "std_accs": std_accs, "fold_accs": fold_accs_all,
                "fit": fit,
            })

        with open(cache_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"\n  Cached → {cache_path}")

    # ── Step 2: Arrhenius fit ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ARRHENIUS FIT")
    print(f"{'='*60}")
    taus     = [r["fit"]["tau"]     for r in all_results]
    tau_errs = [r["fit"]["tau_err"] for r in all_results]
    temps    = [r["T"]              for r in all_results]
    arr      = fit_arrhenius(temps, taus, tau_errs)

    with open(os.path.join(args.out_dir, "arrhenius_results.pkl"), "wb") as f:
        pickle.dump({"forgetting_curves": all_results, "arrhenius": arr}, f)

    # ── Step 3: Figures ───────────────────────────────────────────────────
    plot_forgetting_curves(all_results,
        os.path.join(args.out_dir, "fig_forgetting_curves.png"))
    plot_arrhenius(all_results, arr,
        os.path.join(args.out_dir, "fig_arrhenius.png"))

    # ── Step 4: Summary ───────────────────────────────────────────────────
    rows = []
    for r in all_results:
        f = r["fit"]
        rows.append({
            "T (LJ)": r["T"], "T/Tg": r["T_Tg"],
            "1/T":    round(1/r["T"], 4),
            "A0 (%)": f"{f['A0']:.1f}±{f['A0_err']:.1f}",
            "A_inf (%)": f"{f['A_inf']:.1f}±{f['A_inf_err']:.1f}",
            "tau":    f"{f['tau']:.1f}±{f['tau_err']:.1f}",
            "beta":   f"{f['beta']:.3f}±{f['beta_err']:.3f}",
            "fit_R2": f"{f['r2']:.4f}",
            "converged": f["converged"],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "arrhenius_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  FORGETTING CURVE SUMMARY")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    Ea = arr["Ea"]; Ea_err = arr["Ea_err"]
    print(f"\n{'='*60}")
    print(f"  ARRHENIUS RESULT")
    print(f"{'='*60}")
    print(f"  Ea    = {Ea:.3f} ± {Ea_err:.3f} ε  (LJ reduced units)")
    print(f"  τ0    = {arr['tau0']:.2f} cycles")
    print(f"  R²    = {arr['r2']:.4f}")
    print(f"  p     = {arr['p']:.4f}")
    print(f"  n_pts = {arr['n_points']}")
    within = MCT_EA_LO <= Ea <= MCT_EA_HI
    if within:
        print(f"\n  ✓ Ea is within the MCT cooperative rearrangement barrier")
        print(f"    range [{MCT_EA_LO}, {MCT_EA_HI}] ε for LJ glass at ρ=1.2")
        print(f"  → The GNN structural probe measures the same physical process")
        print(f"    as classical glass relaxation theory.  This is the key")
        print(f"    result that connects ML structural probes to the PEL.")
    else:
        print(f"\n  ⚠ Ea = {Ea:.2f} ε is outside the expected MCT range.")
        print(f"    Possible causes: finite-size effects (N=256), short")
        print(f"    cycle protocol (400 cycles < alpha-relaxation timescale),")
        print(f"    or genuine deviation from Arrhenius behaviour.")

    print(f"\n  Done. All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()