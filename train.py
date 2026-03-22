#!/usr/bin/env python3
"""
==========================================================================
MULTI-HISTORY GLASS: TRAINING & VALIDATION
Paper: "Geometric Orthogonality of Competing Structural Memories in Glass"
==========================================================================

EXPERIMENTS
─────────────────────────────────────────────────────────────────────────
Exp 1  Single-task baselines
       1A: Cooling-only classifier  (binary: fast vs slow)
       1B: Fatigue-only classifier  (3-class: c0 vs c200 vs c400)

Exp 2  Multi-task classification  (CORE)
       Simultaneous prediction of both cooling + fatigue labels
       from a single graph.  Key question: does joint training
       hurt either task vs single-task?  Answer = degree of
       geometric orthogonality.

Exp 3  Latent space orthogonality analysis  (CORE SCIENTIFIC RESULT)
       Extract 64-D latent vectors from trained multi-task model.
       PCA projection → Spearman ρ(PC1, cooling) vs ρ(PC2, fatigue).
       If ρ_cooling >> ρ_fatigue on PC1 AND ρ_fatigue >> ρ_cooling
       on PC2 → histories occupy geometrically orthogonal subspaces.

Exp 4  History dominance test
       Train cooling classifier on FATIGUED-only subset (cycle 400).
       Compare accuracy to training on pristine subset (cycle 0).
       If accuracy drops → fatigue partially overwrites thermal memory.
       If accuracy holds → geometric orthogonality confirmed.

Exp 5  Continuous fatigue regression  (5-fold CV)
       Predict normalised cycle number from structure.
       Trained separately for fast vs slow glasses to test whether
       thermal history modulates the learnability of fatigue history.

Exp 6  Feature permutation importance per task
       Which of the 8 node features matter for cooling detection?
       Which for fatigue detection?  Are they the same or different?
       Different feature rankings → different structural subspaces.

==========================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import os, sys, time, gc, warnings, pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (roc_auc_score, f1_score,
                             accuracy_score, r2_score,
                             confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ──────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

# ── Dataset paths ─────────────────────────────────────────────────────────
DATA_PATH = r"D:\final_exp\paper_v3\multihistory_data\multihistory_glasses_amp_8_t_44.pkl"
OUT_DIR   = "multihistory_results_amp_8_t_44"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Physical parameters (must match datagen script) ───────────────────────
BOX_L     = float((256 / 1.2) ** (1.0 / 3.0))   # ≈ 5.975 σ
RC_GRAPH  = 1.5                                  # first shell cutoff
N_CYCLES  = 400                                  # max fatigue cycles

# ── Feature extraction ────────────────────────────────────────────────────
N_FEAT         = 8        # 8D features: r̄, σr, rmin, rmax, d̃, skew, Q25, Q75
INSTANCE_NORM  = True     # per-graph z-score normalisation  (always ON)
FEAT_NAMES     = ["r_mean", "r_std", "r_min", "r_max",
                  "coord_norm", "skewness", "Q25", "Q75"]

# ── GNN architecture ──────────────────────────────────────────────────────
HIDDEN_DIM = 64
N_HEADS    = 4
LATENT_DIM = 64   # dimension of the shared representation before output heads

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
LR          = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 150
PATIENCE    = 25
N_FOLDS     = 5
SEED        = 42

# ── Multi-task loss weights ────────────────────────────────────────────────
#   Both tasks are equally weighted.  This is the neutral assumption.
#   Changing these reveals which task "dominates" gradient flow — another
#   measurable quantity related to the orthogonality hypothesis.
W_COOLING = 1.0
W_FATIGUE = 1.0

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DataLoader workers ─────────────────────────────────────────────────────
# Windows uses spawn-based multiprocessing which re-imports the module for
# every worker process, printing the header multiple times and sometimes
# deadlocking.  Setting num_workers=0 disables multiprocessing entirely —
# data loading runs in the main process.  On Linux/macOS fork is used and
# num_workers=2 is safe and faster.
import platform as _platform
NUM_WORKERS = 0 if _platform.system() == "Windows" else 2

def _print_header():
    print(f"\n{'='*60}")
    print(f"  MULTI-HISTORY GLASS TRAINING")
    print(f"{'─'*60}")
    print(f"  Device      : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"  Features    : {N_FEAT}D  (instance_norm={INSTANCE_NORM})")
    print(f"  Folds       : {N_FOLDS}")
    print(f"  Max epochs  : {MAX_EPOCHS}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────
# 2.  FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────

def extract_features(positions: np.ndarray,
                     box:        float   = BOX_L,
                     rc:         float   = RC_GRAPH,
                     instance_norm: bool = INSTANCE_NORM
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    8D local bond-length feature extraction with minimum-image PBC.

    Features per atom i (first shell, r < rc):
      0  r̄_i       mean bond length
      1  σr_i      bond-length standard deviation
      2  rmin_i    minimum bond length
      3  rmax_i    maximum bond length
      4  d̃_i       normalised coordination number
      5  γ̃_i       skewness proxy  (clipped to ±5)
      6  Q25_i     25th percentile of bond lengths
      7  Q75_i     75th percentile of bond lengths

    Instance normalisation (per-graph z-score) is applied if requested.
    This removes global magnitude shifts, forcing the GNN to use only
    relational geometry.

    Returns
    -------
    node_feat  : (N, 8) float32
    edge_index : (2, E) int64
    edge_attr  : (E, 1) float32  — raw bond lengths (not normalised)
    """
    N   = len(positions)
    pos = np.asarray(positions, dtype=np.float32)

    # Pairwise distances with minimum-image PBC
    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box * np.round(dr / box)
    dist = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))           # (N, N)
    nbr  = (dist > 1e-6) & (dist < rc)

    # Per-atom features
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

    # Instance normalisation
    if instance_norm:
        mu_g  = feats.mean(axis=0, keepdims=True)
        sig_g = feats.std(axis=0,  keepdims=True) + 1e-8
        feats = (feats - mu_g) / sig_g

    # Edge construction
    rows, cols = np.where(nbr)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_attr  = dist[rows, cols].reshape(-1, 1).astype(np.float32)

    return feats, edge_index, edge_attr


# ──────────────────────────────────────────────────────────────────────────
# 3.  DATASET BUILDING
# ──────────────────────────────────────────────────────────────────────────

def load_and_build_graphs(data_path: str) -> List[Data]:
    """
    Load raw pickle dataset and convert every sample to a PyG Data object.

    Each Data object carries:
      x            : (256, 8) node features
      edge_index   : (2, E)
      edge_attr    : (E, 1)
      y_cooling    : int  0/1     (binary: fast / slow)
      y_fatigue    : int  0/1/2   (3-class: cycle 0 / 200 / 400)
      y_fatigue_cont : float [0,1] (regression target)
      glass_id     : int
      cooling_type : str
      fatigue_cycles : int
      combined_label : int  0–5   (for stratified CV)
    """
    with open(data_path, "rb") as f:
        raw = pickle.load(f)

    print(f"Loaded {len(raw)} raw samples from {data_path}")
    graphs, n_nan = [], 0
    t0 = time.time()

    for s in raw:
        nf, ei, ea = extract_features(s["positions"])
        if not (np.isfinite(nf).all() and np.isfinite(ea).all()):
            n_nan += 1
            continue

        d = Data(
            x            = torch.from_numpy(nf).float(),
            edge_index   = torch.from_numpy(ei).long(),
            edge_attr    = torch.from_numpy(ea).float(),
            y_cooling    = torch.tensor(s["label_cooling"],      dtype=torch.long),
            y_fatigue    = torch.tensor(s["label_fatigue"],      dtype=torch.long),
            y_fatigue_cont = torch.tensor([s["label_fatigue_cont"]], dtype=torch.float),
            glass_id     = s["glass_id"],
            cooling_type = s["cooling_type"],
            fatigue_cycles = s["fatigue_cycles"],
            # Combined label for stratified splitting: 0–5
            combined_label = s["label_cooling"] * 3 + s["label_fatigue"],
        )
        graphs.append(d)

    print(f"  Built {len(graphs)} graphs  "
          f"({n_nan} NaN skipped)  in {time.time()-t0:.1f} s")
    return graphs


# ──────────────────────────────────────────────────────────────────────────
# 4.  MODEL ARCHITECTURES
# ──────────────────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    Common backbone for all models:
    Linear encoder  →  GATv2 layer 1  →  GATv2 layer 2
    →  global mean pool  →  post-MP MLP  →  64-D latent vector z.
    """
    def __init__(self, in_dim: int = N_FEAT):
        super().__init__()
        self.enc  = nn.Linear(in_dim, HIDDEN_DIM)
        self.gat1 = GATv2Conv(HIDDEN_DIM, HIDDEN_DIM,
                              heads=N_HEADS, edge_dim=1, concat=True)
        self.gat2 = GATv2Conv(HIDDEN_DIM * N_HEADS, HIDDEN_DIM,
                              heads=N_HEADS, edge_dim=1, concat=True)
        pool_dim  = HIDDEN_DIM * N_HEADS
        self.post = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, LATENT_DIM),
            nn.ReLU(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, ei, ea, batch = (data.x, data.edge_index,
                            data.edge_attr, data.batch)
        x = F.relu(self.enc(x))
        x = F.relu(self.gat1(x, ei, ea))
        x = F.relu(self.gat2(x, ei, ea))
        x = global_mean_pool(x, batch)      # (B, pool_dim)
        return self.post(x)                 # (B, LATENT_DIM)


class GATv2MultiTask(nn.Module):
    """
    Multi-task classifier.

    Outputs
    -------
    logit_cooling : (B,)    BCEWithLogitsLoss  (fast vs slow)
    logit_fatigue : (B, 3)  CrossEntropyLoss   (c0 / c200 / c400)
    """
    def __init__(self, in_dim: int = N_FEAT):
        super().__init__()
        self.encoder      = SharedEncoder(in_dim)
        self.head_cooling = nn.Linear(LATENT_DIM, 1)
        self.head_fatigue = nn.Linear(LATENT_DIM, 3)

    def forward(self, data: Data
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z             = self.encoder(data)
        logit_cooling = self.head_cooling(z).squeeze(-1)
        logit_fatigue = self.head_fatigue(z)
        return logit_cooling, logit_fatigue, z          # also return latent

    def encode(self, data: Data) -> torch.Tensor:
        """Return latent vectors only (used for orthogonality analysis)."""
        with torch.no_grad():
            z = self.encoder(data)
        return z


class GATv2SingleTask(nn.Module):
    """
    Single-task classifier.
    n_classes = 1  → binary (BCEWithLogitsLoss)
    n_classes > 1  → multiclass (CrossEntropyLoss)
    """
    def __init__(self, in_dim: int = N_FEAT, n_classes: int = 1):
        super().__init__()
        self.encoder   = SharedEncoder(in_dim)
        self.head      = nn.Linear(LATENT_DIM, max(1, n_classes))
        self.n_classes = n_classes

    def forward(self, data: Data) -> torch.Tensor:
        z = self.encoder(data)
        out = self.head(z)
        if self.n_classes == 1:
            return out.squeeze(-1)
        return out


class GATv2Regressor(nn.Module):
    """Single-output regression: predict normalised cycle number ∈ [0,1]."""
    def __init__(self, in_dim: int = N_FEAT):
        super().__init__()
        self.encoder = SharedEncoder(in_dim)
        self.head    = nn.Linear(LATENT_DIM, 1)

    def forward(self, data: Data) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(data))).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
# 5.  TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────────

def make_optimizer(model: nn.Module) -> Tuple:
    opt  = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)
    scl  = GradScaler(enabled=(DEVICE.type == "cuda"))
    return opt, sch, scl


# ── Multi-task ────────────────────────────────────────────────────────────

def multitask_loss(logit_c: torch.Tensor,
                   logit_f: torch.Tensor,
                   y_c:     torch.Tensor,
                   y_f:     torch.Tensor) -> torch.Tensor:
    l_cool = F.binary_cross_entropy_with_logits(logit_c, y_c.float())
    l_fat  = F.cross_entropy(logit_f, y_f)
    return W_COOLING * l_cool + W_FATIGUE * l_fat


def train_multitask_epoch(model, loader, opt, scl):
    model.train()
    total = 0.0
    for b in loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            lc, lf, _ = model(b)
            loss = multitask_loss(lc, lf, b.y_cooling, b.y_fatigue)
        scl.scale(loss).backward()
        scl.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scl.step(opt)
        scl.update()
        total += loss.item() * b.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_multitask(model, loader):
    model.eval()
    all_lc, all_lf, all_yc, all_yf, all_z = [], [], [], [], []
    for b in loader:
        b = b.to(DEVICE)
        lc, lf, z = model(b)
        all_lc.append(lc.cpu()); all_lf.append(lf.cpu())
        all_yc.append(b.y_cooling.cpu()); all_yf.append(b.y_fatigue.cpu())
        all_z.append(z.cpu())
    lc = torch.cat(all_lc); lf = torch.cat(all_lf)
    yc = torch.cat(all_yc); yf = torch.cat(all_yf)
    z  = torch.cat(all_z)

    loss = multitask_loss(lc, lf, yc, yf).item()

    # Cooling accuracy & AUC
    pc   = torch.sigmoid(lc).numpy()
    pred_c = (pc > 0.5).astype(int)
    acc_c  = (pred_c == yc.numpy()).mean() * 100
    auc_c  = roc_auc_score(yc.numpy(), pc)

    # Fatigue accuracy
    pred_f = lf.argmax(dim=1).numpy()
    acc_f  = (pred_f == yf.numpy()).mean() * 100
    f1_f   = f1_score(yf.numpy(), pred_f, average="macro")

    return loss, acc_c, auc_c, acc_f, f1_f, z.numpy(), yc.numpy(), yf.numpy()


# ── Single-task binary ────────────────────────────────────────────────────

def train_singletask_epoch(model, loader, opt, scl, n_classes):
    model.train()
    total = 0.0
    for b in loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            out  = model(b)
            y    = b.y_cooling if n_classes == 1 else b.y_fatigue
            loss = (F.binary_cross_entropy_with_logits(out, y.float())
                    if n_classes == 1
                    else F.cross_entropy(out, y))
        scl.scale(loss).backward()
        scl.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scl.step(opt)
        scl.update()
        total += loss.item() * b.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_singletask(model, loader, n_classes):
    model.eval()
    all_out, all_y = [], []
    for b in loader:
        b = b.to(DEVICE)
        out = model(b)
        y   = b.y_cooling if n_classes == 1 else b.y_fatigue
        all_out.append(out.cpu())
        all_y.append(y.cpu())
    out = torch.cat(all_out); y = torch.cat(all_y)
    y_np  = y.numpy()

    if n_classes == 1:
        loss = F.binary_cross_entropy_with_logits(out, y.float()).item()
        prob = torch.sigmoid(out).numpy()
        pred = (prob > 0.5).astype(int)
        acc  = (pred == y_np).mean() * 100
        auc  = roc_auc_score(y_np, prob)
        return loss, acc, auc, None
    else:
        loss = F.cross_entropy(out, y).item()
        pred = out.argmax(dim=1).numpy()
        acc  = (pred == y_np).mean() * 100
        f1   = f1_score(y_np, pred, average="macro")
        return loss, acc, None, f1


# ── Regression ────────────────────────────────────────────────────────────

def train_reg_epoch(model, loader, opt, scl):
    model.train()
    total = 0.0
    for b in loader:
        b = b.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred = model(b)
            loss = F.mse_loss(pred, b.y_fatigue_cont.squeeze(-1))
        scl.scale(loss).backward()
        scl.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scl.step(opt)
        scl.update()
        total += loss.item() * b.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_reg(model, loader):
    model.eval()
    all_pred, all_y = [], []
    for b in loader:
        b = b.to(DEVICE)
        all_pred.append(model(b).cpu())
        all_y.append(b.y_fatigue_cont.squeeze(-1).cpu())
    pred = torch.cat(all_pred).numpy()
    y    = torch.cat(all_y).numpy()
    return r2_score(y, pred), float(((pred - y)**2).mean())


# ── Generic CV loop ───────────────────────────────────────────────────────

def run_cv(graphs:     List[Data],
           strat_labels: np.ndarray,
           model_fn,           # callable() → model
           train_fn,           # (model, loader, opt, scl) → loss
           eval_fn,            # (model, loader) → (loss, metric, ...)
           metric_name: str,
           tag:         str,
           max_epochs:  int  = MAX_EPOCHS,
           patience:    int  = PATIENCE) -> List[Dict]:
    """
    Generic 5-fold stratified cross-validation runner.
    Stratification uses strat_labels (combined 0–5 for multi-task,
    or the specific task label for single-task experiments).
    """
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = []

    for fi, (tr_idx, va_idx) in enumerate(skf.split(strat_labels, strat_labels)):
        tr_loader = DataLoader([graphs[i] for i in tr_idx],
                               batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS)
        va_loader = DataLoader([graphs[i] for i in va_idx],
                               batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS)

        model = model_fn().to(DEVICE)
        opt, sch, scl = make_optimizer(model)

        best_loss  = float("inf")
        best_res   = None
        patience_ctr = 0

        for ep in range(1, max_epochs + 1):
            train_fn(model, tr_loader, opt, scl)
            res  = eval_fn(model, va_loader)
            loss = res[0]
            sch.step()

            if loss < best_loss:
                best_loss = loss
                best_res  = res
                best_ep   = ep
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        results.append({"fold": fi+1, "result": best_res, "epoch": best_ep})
        metric_val = best_res[1] if best_res else float("nan")
        print(f"    fold {fi+1}: {metric_name}={metric_val:.2f}  "
              f"(ep {best_ep})")

    return results


# ──────────────────────────────────────────────────────────────────────────
# 6.  EXPERIMENT RUNNERS
# ──────────────────────────────────────────────────────────────────────────

# ── Exp 1A & 1B: Single-task baselines ────────────────────────────────────

def exp_singletask(graphs: List[Data]) -> Dict:
    """
    1A: cooling classifier (binary)
    1B: fatigue classifier (3-class)
    Both use 5-fold stratified CV on their respective labels.
    """
    results = {}

    # ── 1A: cooling ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EXP 1A: Single-task COOLING classifier")
    print(f"{'─'*60}")
    lbl_c = np.array([g.y_cooling.item() for g in graphs])

    def make_cooling():
        return GATv2SingleTask(n_classes=1)

    def train_c(m, ld, o, s):
        return train_singletask_epoch(m, ld, o, s, n_classes=1)

    def eval_c(m, ld):
        return eval_singletask(m, ld, n_classes=1)

    r1a = run_cv(graphs, lbl_c, make_cooling, train_c, eval_c,
                 "acc_cool", "1A")
    accs_c = [r["result"][1] for r in r1a]
    aucs_c = [r["result"][2] for r in r1a]
    print(f"\n  1A COOLING:  "
          f"acc={np.mean(accs_c):.2f}±{np.std(accs_c):.2f}%  "
          f"AUC={np.mean(aucs_c):.4f}±{np.std(aucs_c):.4f}")
    results["1A_cooling"] = {"accs": accs_c, "aucs": aucs_c}

    # ── 1B: fatigue ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EXP 1B: Single-task FATIGUE classifier (3-class)")
    print(f"{'─'*60}")
    lbl_f = np.array([g.y_fatigue.item() for g in graphs])

    def make_fatigue():
        return GATv2SingleTask(n_classes=3)

    def train_f(m, ld, o, s):
        return train_singletask_epoch(m, ld, o, s, n_classes=3)

    def eval_f(m, ld):
        return eval_singletask(m, ld, n_classes=3)

    r1b = run_cv(graphs, lbl_f, make_fatigue, train_f, eval_f,
                 "acc_fat", "1B")
    accs_f = [r["result"][1] for r in r1b]
    f1s_f  = [r["result"][3] for r in r1b]
    print(f"\n  1B FATIGUE:  "
          f"acc={np.mean(accs_f):.2f}±{np.std(accs_f):.2f}%  "
          f"F1={np.mean(f1s_f):.4f}±{np.std(f1s_f):.4f}")
    results["1B_fatigue"] = {"accs": accs_f, "f1s": f1s_f}

    return results


# ── Exp 2: Multi-task + Exp 3: Latent space orthogonality ─────────────────

def exp_multitask_and_orthogonality(graphs: List[Data]) -> Dict:
    """
    Train GATv2MultiTask under 5-fold CV.
    After training, extract latent vectors for orthogonality analysis.
    """
    print(f"\n{'─'*60}")
    print(f"  EXP 2: Multi-task classification (CORE)")
    print(f"  EXP 3: Latent space orthogonality (CORE RESULT)")
    print(f"{'─'*60}")

    combined_lbl = np.array([g.combined_label for g in graphs])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results  = []
    all_z, all_yc, all_yf = [], [], []   # collect from all val folds

    for fi, (tr_idx, va_idx) in enumerate(skf.split(combined_lbl,
                                                      combined_lbl)):
        tr_loader = DataLoader([graphs[i] for i in tr_idx],
                               batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS)
        va_loader = DataLoader([graphs[i] for i in va_idx],
                               batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS)

        model = GATv2MultiTask().to(DEVICE)
        opt, sch, scl = make_optimizer(model)

        best_loss  = float("inf")
        best_res   = None
        patience_ctr = 0

        for ep in range(1, MAX_EPOCHS + 1):
            train_multitask_epoch(model, tr_loader, opt, scl)
            res  = eval_multitask(model, va_loader)
            loss = res[0]
            sch.step()

            if loss < best_loss:
                best_loss = loss
                best_res  = res
                best_ep   = ep
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        _, acc_c, auc_c, acc_f, f1_f, z_val, yc_val, yf_val = best_res
        fold_results.append({
            "fold": fi+1, "acc_c": acc_c, "auc_c": auc_c,
            "acc_f": acc_f, "f1_f": f1_f, "epoch": best_ep
        })

        # Accumulate latent vectors across all val folds
        all_z.append(z_val)
        all_yc.append(yc_val)
        all_yf.append(yf_val)

        print(f"    fold {fi+1}: cool={acc_c:.1f}%(AUC={auc_c:.4f})  "
              f"fat={acc_f:.1f}%(F1={f1_f:.4f})  ep={best_ep}")

    # ── Summary ───────────────────────────────────────────────────────────
    accs_c = [r["acc_c"] for r in fold_results]
    aucs_c = [r["auc_c"] for r in fold_results]
    accs_f = [r["acc_f"] for r in fold_results]
    f1s_f  = [r["f1_f"]  for r in fold_results]

    print(f"\n  MULTI-TASK RESULTS (5-fold CV)")
    print(f"  Cooling : acc={np.mean(accs_c):.2f}±{np.std(accs_c):.2f}%  "
          f"AUC={np.mean(aucs_c):.4f}±{np.std(aucs_c):.4f}")
    print(f"  Fatigue : acc={np.mean(accs_f):.2f}±{np.std(accs_f):.2f}%  "
          f"F1={np.mean(f1s_f):.4f}±{np.std(f1s_f):.4f}")

    # ── Orthogonality analysis ─────────────────────────────────────────────
    Z  = np.vstack(all_z)      # (600, 64)
    YC = np.concatenate(all_yc)
    YF = np.concatenate(all_yf)

    orth_result = orthogonality_analysis(Z, YC, YF)

    return {
        "fold_results": fold_results,
        "accs_c": accs_c, "aucs_c": aucs_c,
        "accs_f": accs_f, "f1s_f":  f1s_f,
        "orthogonality": orth_result,
        "latent_Z": Z, "latent_YC": YC, "latent_YF": YF,
    }


# ── Exp 3: Orthogonality analysis (standalone) ────────────────────────────

def orthogonality_analysis(Z: np.ndarray,
                           YC: np.ndarray,
                           YF: np.ndarray) -> Dict:
    """
    Test geometric orthogonality of cooling and fatigue histories.

    Method
    ──────
    1. Standardise Z and project to 10 PCs.
    2. Compute Spearman ρ(PCk, YC) and ρ(PCk, YF) for k=1..10.
    3. Orthogonality criterion:
       - PC1 dominated by ONE label (ρ > 0.4 for one, < 0.15 for other)
       - PC2 dominated by the OTHER label
    4. Off-diagonal cross-correlations should be small.
    5. Mutual information lower bounds via Spearman ρ.

    Returns
    -------
    Dict with PCA explained variance, Spearman correlations,
    orthogonality score, and interpretation string.
    """
    scaler = StandardScaler()
    Z_norm = scaler.fit_transform(Z)

    pca = PCA(n_components=10)
    Z_pca = pca.fit_transform(Z_norm)          # (N, 10)

    evr = pca.explained_variance_ratio_

    rho_c = []   # Spearman ρ(PCk, cooling label)
    rho_f = []   # Spearman ρ(PCk, fatigue label)
    for k in range(10):
        rc, _ = spearmanr(Z_pca[:, k], YC)
        rf, _ = spearmanr(Z_pca[:, k], YF)
        rho_c.append(float(rc))
        rho_f.append(float(rf))

    # Best PC for each task (highest |ρ|)
    best_c = int(np.argmax(np.abs(rho_c)))
    best_f = int(np.argmax(np.abs(rho_f)))

    print(f"\n  LATENT SPACE ORTHOGONALITY ANALYSIS")
    print(f"  {'PC':>4}  {'Var%':>6}  {'ρ(cool)':>9}  {'ρ(fat)':>9}  "
          f"{'Dominant':>12}")
    print(f"  {'─'*50}")
    for k in range(10):
        dom = ("cooling" if abs(rho_c[k]) > abs(rho_f[k])
               else "fatigue")
        mark_c = " ←" if k == best_c else ""
        mark_f = " ←" if k == best_f else ""
        print(f"  {k+1:>4}  {evr[k]*100:>5.1f}%  "
              f"{rho_c[k]:>+8.4f}{mark_c}  "
              f"{rho_f[k]:>+8.4f}{mark_f}  "
              f"{dom:>12}")

    # Orthogonality score: how different are the two best PCs?
    orthogonal = (best_c != best_f)
    cross_c_on_best_f = abs(rho_c[best_f])   # cooling signal on fatigue's PC
    cross_f_on_best_c = abs(rho_f[best_c])   # fatigue signal on cooling's PC
    cross_contamination = (cross_c_on_best_f + cross_f_on_best_c) / 2

    print(f"\n  Best PC for cooling : PC{best_c+1}  (|ρ|={abs(rho_c[best_c]):.4f})")
    print(f"  Best PC for fatigue : PC{best_f+1}  (|ρ|={abs(rho_f[best_f]):.4f})")
    print(f"  Different PCs       : {orthogonal}")
    print(f"  Cross-contamination : {cross_contamination:.4f}  "
          f"(cooling signal on fatigue PC + vice versa, halved)")

    if orthogonal and cross_contamination < 0.15:
        interpretation = ("STRONG ORTHOGONALITY — histories occupy distinct "
                          "latent directions; simultaneous decoding is lossless")
    elif orthogonal and cross_contamination < 0.30:
        interpretation = ("PARTIAL ORTHOGONALITY — distinct principal axes "
                          "but moderate cross-contamination; some interference")
    else:
        interpretation = ("SHARED SUBSPACE — both histories encoded along "
                          "similar latent directions; strong interference")

    print(f"\n  INTERPRETATION: {interpretation}")

    return {
        "explained_variance": evr.tolist(),
        "rho_cooling": rho_c,
        "rho_fatigue": rho_f,
        "best_pc_cooling": best_c,
        "best_pc_fatigue": best_f,
        "orthogonal": orthogonal,
        "cross_contamination": cross_contamination,
        "interpretation": interpretation,
        "Z_pca": Z_pca,
        "YC": YC,
        "YF": YF,
    }


# ── Exp 4: History dominance test ─────────────────────────────────────────

def exp_history_dominance(graphs: List[Data]) -> Dict:
    """
    Test whether fatigue erases thermal (cooling) memory.

    Three sub-experiments:
    4A: Classify cooling on PRISTINE glasses  (cycle 0 only)
    4B: Classify cooling on FATIGUED glasses  (cycle 400 only)
    4C: Classify cooling on ALL fatigued     (cycles 200+400)

    If 4A_acc >> 4B_acc: fatigue overwrites thermal memory.
    If 4A_acc ≈ 4B_acc:  histories are geometrically orthogonal.
    """
    print(f"\n{'─'*60}")
    print(f"  EXP 4: History Dominance Test")
    print(f"  Does fatigue erase thermal (cooling-rate) memory?")
    print(f"{'─'*60}")

    def _run_cooling_subset(subset_tag: str,
                            cycle_filter) -> Tuple[float, float]:
        sub = [g for g in graphs if g.fatigue_cycles in cycle_filter]
        if len(sub) < 10:
            print(f"  WARNING: {subset_tag} subset too small ({len(sub)})")
            return float("nan"), float("nan")
        lbl = np.array([g.y_cooling.item() for g in sub])
        unique, counts = np.unique(lbl, return_counts=True)
        print(f"  {subset_tag}: {len(sub)} glasses  "
              f"(fast={counts[0] if 0 in unique else 0}, "
              f"slow={counts[1] if 1 in unique else 0})")

        def make():
            return GATv2SingleTask(n_classes=1)

        def train_(m, ld, o, s):
            return train_singletask_epoch(m, ld, o, s, n_classes=1)

        def eval_(m, ld):
            return eval_singletask(m, ld, n_classes=1)

        rs = run_cv(sub, lbl, make, train_, eval_, "acc_cool", subset_tag)
        accs = [r["result"][1] for r in rs]
        aucs = [r["result"][2] for r in rs]
        print(f"  {subset_tag}: acc={np.mean(accs):.2f}±{np.std(accs):.2f}%  "
              f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")
        return np.mean(accs), np.mean(aucs)

    acc_pristine, auc_pristine = _run_cooling_subset(
        "4A pristine (cycle 0)",  {0})
    acc_fat400,   auc_fat400   = _run_cooling_subset(
        "4B fatigued (cycle 400)", {400})
    acc_fat_both, auc_fat_both = _run_cooling_subset(
        "4C fatigued (200+400)",   {200, 400})

    drop_4b = acc_pristine - acc_fat400
    drop_4c = acc_pristine - acc_fat_both

    print(f"\n  DOMINANCE SUMMARY")
    print(f"  acc(pristine)   = {acc_pristine:.2f}%")
    print(f"  acc(fat cycle400) = {acc_fat400:.2f}%  "
          f"(drop = {drop_4b:+.2f} pp)")
    print(f"  acc(fat 200+400)  = {acc_fat_both:.2f}%  "
          f"(drop = {drop_4c:+.2f} pp)")

    if drop_4b < 3.0:
        dom_interp = ("Thermal memory PERSISTS through fatigue — "
                      "cooling history is stored in a geometrically "
                      "orthogonal subspace to fatigue damage")
    elif drop_4b < 8.0:
        dom_interp = ("Moderate erosion — fatigue partially overwrites "
                      "thermal memory; histories share some structural degrees of freedom")
    else:
        dom_interp = ("Thermal memory is ERASED by fatigue — "
                      "fatigue dominates the bond-length geometry")

    print(f"\n  INTERPRETATION: {dom_interp}")

    return {
        "acc_pristine": acc_pristine, "auc_pristine": auc_pristine,
        "acc_fat400":   acc_fat400,   "auc_fat400":   auc_fat400,
        "acc_fat_both": acc_fat_both, "auc_fat_both": auc_fat_both,
        "drop_4b": drop_4b, "drop_4c": drop_4c,
        "interpretation": dom_interp,
    }


# ── Exp 5: Continuous fatigue regression ──────────────────────────────────

def exp_regression(graphs: List[Data]) -> Dict:
    """
    Predict normalised cycle number ∈ [0,1] separately for fast and slow glasses.
    If R²_fast ≠ R²_slow, the cooling history modulates the learnability of fatigue.
    This would mean the two histories are not fully independent — a subtle interference.
    """
    print(f"\n{'─'*60}")
    print(f"  EXP 5: Continuous fatigue regression")
    print(f"  Fast and slow glasses trained and evaluated separately")
    print(f"{'─'*60}")

    results = {}

    for ct, ct_lbl in [("fast", "5A"), ("slow", "5B"), ("all", "5C")]:
        if ct == "all":
            sub = graphs
        else:
            sub = [g for g in graphs if g.cooling_type == ct]

        # Use fatigue label for stratification in regression
        strat = np.array([g.y_fatigue.item() for g in sub])

        def make():
            return GATv2Regressor()

        def train_(m, ld, o, s):
            return train_reg_epoch(m, ld, o, s)

        def eval_(m, ld):
            r2, mse = eval_reg(m, ld)
            return mse, r2 * 100   # return (loss=mse, metric=R²*100)

        print(f"\n  [{ct_lbl}] {ct.upper()} glasses ({len(sub)} samples)")
        rs = run_cv(sub, strat, make, train_, eval_, "R²%", ct_lbl)
        r2s = [r["result"][1] / 100 for r in rs]   # convert back
        mses = [r["result"][0] for r in rs]

        print(f"  {ct_lbl} {ct}: R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}  "
              f"MSE={np.mean(mses):.4f}±{np.std(mses):.4f}")
        results[ct_lbl] = {"r2s": r2s, "mses": mses, "cooling_type": ct}

    # Interpret R² difference between fast and slow
    r2_fast = np.mean(results["5A"]["r2s"])
    r2_slow = np.mean(results["5B"]["r2s"])
    diff    = r2_fast - r2_slow
    print(f"\n  REGRESSION COMPARISON")
    print(f"  R²(fast) = {r2_fast:.4f}")
    print(f"  R²(slow) = {r2_slow:.4f}")
    print(f"  Difference = {diff:+.4f}")
    if abs(diff) < 0.05:
        print(f"  → Fatigue trajectory equally learnable for fast and slow glasses")
        print(f"    (histories are independent)")
    elif diff > 0:
        print(f"  → Fatigue MORE learnable in fast glasses "
              f"(larger structural change per cycle from high-disorder start)")
    else:
        print(f"  → Fatigue MORE learnable in slow glasses "
              f"(more uniform starting point)")

    return results


# ── Exp 6: Permutation importance per task ────────────────────────────────

def exp_permutation_importance(graphs: List[Data]) -> Dict:
    """
    Train separate cooling and fatigue classifiers.
    For each model and each feature, zero out that column across all nodes
    and measure the AUC drop.  Compare rankings between the two tasks.

    If rankings differ: the two tasks use different structural information
    → geometrically orthogonal encoding.
    If rankings are similar: both tasks use the same features → shared subspace.
    """
    print(f"\n{'─'*60}")
    print(f"  EXP 6: Permutation importance per task")
    print(f"{'─'*60}")

    def _train_full_model(graphs_sub, n_classes):
        """Train one model on 80% of data, return model + 20% val set."""
        lbl = (np.array([g.y_cooling.item() for g in graphs_sub])
               if n_classes == 1
               else np.array([g.y_fatigue.item() for g in graphs_sub]))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr_idx, va_idx = next(iter(skf.split(lbl, lbl)))

        tr_loader = DataLoader([graphs_sub[i] for i in tr_idx],
                               batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader([graphs_sub[i] for i in va_idx],
                               batch_size=BATCH_SIZE, shuffle=False)

        model = GATv2SingleTask(n_classes=n_classes).to(DEVICE)
        opt, sch, scl = make_optimizer(model)
        best_loss, best_state = float("inf"), None
        patience_ctr = 0

        for ep in range(1, MAX_EPOCHS + 1):
            train_singletask_epoch(model, tr_loader, opt, scl, n_classes)
            res  = eval_singletask(model, va_loader, n_classes)
            loss = res[0]
            sch.step()
            if loss < best_loss:
                best_loss  = loss
                best_state = {k: v.clone() for k, v in
                              model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        model.load_state_dict(best_state)
        return model, [graphs_sub[i] for i in va_idx], n_classes

    def _auc_zeroed(model, val_graphs, n_classes, feat_k):
        """AUC after zeroing feature k in all validation graphs."""
        zeroed = []
        for g in val_graphs:
            g2 = g.clone()
            g2.x = g2.x.clone()
            g2.x[:, feat_k] = 0.0
            zeroed.append(g2)
        loader = DataLoader(zeroed, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                out = model(b)
                y   = b.y_cooling if n_classes == 1 else b.y_fatigue
                all_out.append(out.cpu())
                all_y.append(y.cpu())
        out = torch.cat(all_out); y_np = torch.cat(all_y).numpy()

        if n_classes == 1:
            prob = torch.sigmoid(out).numpy()
            return float(roc_auc_score(y_np, prob))
        else:
            # For 3-class use macro OVR AUC
            probs = F.softmax(out, dim=1).numpy()
            return float(roc_auc_score(y_np, probs, multi_class="ovr",
                                       average="macro"))

    results = {}

    for task, nc, task_name in [
            ("cooling", 1, "6A cooling"),
            ("fatigue", 3, "6B fatigue")]:

        print(f"\n  [{task_name}]")
        model, val_graphs, nc = _train_full_model(graphs, nc)

        # Intact AUC
        loader_intact = DataLoader(val_graphs, batch_size=BATCH_SIZE,
                                   shuffle=False)
        model.eval()
        all_out, all_y = [], []
        with torch.no_grad():
            for b in loader_intact:
                b = b.to(DEVICE)
                out = model(b)
                y   = b.y_cooling if nc == 1 else b.y_fatigue
                all_out.append(out.cpu()); all_y.append(y.cpu())
        out = torch.cat(all_out); y_np = torch.cat(all_y).numpy()
        if nc == 1:
            auc_intact = float(roc_auc_score(
                y_np, torch.sigmoid(out).numpy()))
        else:
            probs = F.softmax(out, dim=1).numpy()
            auc_intact = float(roc_auc_score(y_np, probs,
                                              multi_class="ovr",
                                              average="macro"))
        print(f"  Intact AUC = {auc_intact:.4f}")

        # Zero-out importance
        drops = []
        for k, fname in enumerate(FEAT_NAMES):
            auc_k = _auc_zeroed(model, val_graphs, nc, k)
            drop  = auc_intact - auc_k
            drops.append(drop)
            print(f"    {fname:<14}: AUC drop = {drop:+.4f}")

        # Sort by importance
        ranked = sorted(zip(drops, FEAT_NAMES), reverse=True)
        print(f"\n  RANKING ({task_name}):")
        for rank, (d, fn) in enumerate(ranked, 1):
            print(f"    {rank}. {fn:<14}  Δ AUC = {d:+.4f}")

        results[task] = {"auc_intact": auc_intact,
                         "drops": drops,
                         "ranked": ranked}

    # Compare rankings between tasks
    print(f"\n  FEATURE RANKING COMPARISON")
    print(f"  {'Feature':<14}  {'Rank(cool)':>10}  {'Rank(fat)':>10}  "
          f"{'Δ rank':>8}")
    cool_rank = {fn: i+1 for i, (_, fn) in
                 enumerate(results["cooling"]["ranked"])}
    fat_rank  = {fn: i+1 for i, (_, fn) in
                 enumerate(results["fatigue"]["ranked"])}
    rank_diffs = []
    for fn in FEAT_NAMES:
        rc = cool_rank.get(fn, 0)
        rf = fat_rank.get(fn,  0)
        d  = abs(rc - rf)
        rank_diffs.append(d)
        print(f"  {fn:<14}  {rc:>10}  {rf:>10}  {d:>8}")

    mean_rank_diff = np.mean(rank_diffs)
    print(f"\n  Mean absolute rank difference: {mean_rank_diff:.2f}")
    if mean_rank_diff >= 2.5:
        print(f"  → DIFFERENT feature rankings per task — structural evidence "
              f"that the two histories use distinct geometric degrees of freedom")
    else:
        print(f"  → SIMILAR feature rankings — both histories primarily encoded "
              f"in the same geometric features")

    results["mean_rank_diff"]  = mean_rank_diff
    results["cool_rank"] = cool_rank
    results["fat_rank"]  = fat_rank
    return results


# ──────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────

def plot_orthogonality(orth: Dict, out_path: str) -> None:
    """
    Four-panel figure for the latent space orthogonality result.

    Panel A: 2D PCA scatter coloured by cooling label
    Panel B: 2D PCA scatter coloured by fatigue label
    Panel C: Spearman |ρ| per PC for each task (bar chart)
    Panel D: Scree plot (explained variance %)
    """
    Z_pca = orth["Z_pca"]
    YC    = orth["YC"]
    YF    = orth["YF"]
    evr   = np.array(orth["explained_variance"])
    rho_c = np.abs(orth["rho_cooling"])
    rho_f = np.abs(orth["rho_fatigue"])

    cool_cmap  = plt.cm.RdBu
    fat_colors = plt.cm.viridis(np.linspace(0, 1, 3))

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.40)

    # A: PCA coloured by cooling
    ax_a = fig.add_subplot(gs[0, :2])
    for lc, name, col in [(0, "Fast", "#e6194b"), (1, "Slow", "#4363d8")]:
        idx = YC == lc
        ax_a.scatter(Z_pca[idx, 0], Z_pca[idx, 1],
                     c=col, label=name, alpha=0.55, s=18, edgecolors="none")
    ax_a.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax_a.set_title("(A) Latent space — Cooling label")
    ax_a.legend(framealpha=0.7, fontsize=9)

    # B: PCA coloured by fatigue
    ax_b = fig.add_subplot(gs[1, :2])
    for lf, name, col in [(0, "c0",   fat_colors[0]),
                           (1, "c200", fat_colors[1]),
                           (2, "c400", fat_colors[2])]:
        idx = YF == lf
        ax_b.scatter(Z_pca[idx, 0], Z_pca[idx, 1],
                     c=[col], label=name, alpha=0.55, s=18, edgecolors="none")
    ax_b.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax_b.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax_b.set_title("(B) Latent space — Fatigue label")
    ax_b.legend(framealpha=0.7, fontsize=9)

    # C: |ρ| per PC
    ax_c = fig.add_subplot(gs[0, 2:])
    x = np.arange(1, 11)
    w = 0.35
    ax_c.bar(x - w/2, rho_c[:10], w, label="Cooling", color="#e6194b",
             alpha=0.8)
    ax_c.bar(x + w/2, rho_f[:10], w, label="Fatigue", color="#4363d8",
             alpha=0.8)
    ax_c.axhline(0.15, ls="--", lw=0.8, c="grey",
                 label="threshold (0.15)")
    ax_c.set_xlabel("Principal component")
    ax_c.set_ylabel("|Spearman ρ|")
    ax_c.set_title("(C) Correlation of each PC with task labels")
    ax_c.set_xticks(x)
    ax_c.set_ylim(0, 1)
    ax_c.legend(fontsize=9, framealpha=0.7)

    # D: Scree plot
    ax_d = fig.add_subplot(gs[1, 2:])
    ax_d.bar(x, evr[:10] * 100, color="#3cb44b", alpha=0.8)
    ax_d.set_xlabel("Principal component")
    ax_d.set_ylabel("Explained variance (%)")
    ax_d.set_title("(D) Scree plot")
    ax_d.set_xticks(x)

    plt.suptitle(
        f"Latent Space Orthogonality of Competing Structural Memories\n"
        f"Best PC for cooling: PC{orth['best_pc_cooling']+1}  |  "
        f"Best PC for fatigue: PC{orth['best_pc_fatigue']+1}  |  "
        f"Cross-contamination: {orth['cross_contamination']:.3f}",
        fontsize=11, y=1.01
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Orthogonality figure → {out_path}")


def plot_results_summary(st_results: Dict,
                         mt_results: Dict,
                         dom_results: Dict,
                         reg_results: Dict,
                         out_path: str) -> None:
    """
    Summary bar chart of all classification accuracies across experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: classification accuracy comparison
    labels  = ["1A Cool\n(single)", "2 Cool\n(multi)",
                "1B Fat\n(single)",  "2 Fat\n(multi)"]
    means   = [np.mean(st_results["1A_cooling"]["accs"]),
               np.mean(mt_results["accs_c"]),
               np.mean(st_results["1B_fatigue"]["accs"]),
               np.mean(mt_results["accs_f"])]
    stds    = [np.std(st_results["1A_cooling"]["accs"]),
               np.std(mt_results["accs_c"]),
               np.std(st_results["1B_fatigue"]["accs"]),
               np.std(mt_results["accs_f"])]
    colors  = ["#e6194b", "#f58231", "#4363d8", "#42d4f4"]

    axes[0].bar(range(4), means, yerr=stds, capsize=5,
                color=colors, alpha=0.85, edgecolor="k", lw=0.7)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Classification: single vs multi-task")
    axes[0].set_ylim(40, 105)
    axes[0].axhline(100, ls="--", lw=0.6, c="grey")
    axes[0].axhline(50,  ls=":",  lw=0.6, c="grey", label="chance (2-class)")
    axes[0].axhline(33.3, ls=":", lw=0.6, c="silver",
                    label="chance (3-class)")
    axes[0].legend(fontsize=8)

    # Middle: history dominance
    d_labels = ["Pristine\n(cycle 0)", "Fatigued\n(cycle 400)",
                "Fatigued\n(200+400)"]
    d_accs   = [dom_results["acc_pristine"],
                dom_results["acc_fat400"],
                dom_results["acc_fat_both"]]
    axes[1].bar(range(3), d_accs, color=["#3cb44b", "#e6194b", "#f58231"],
                alpha=0.85, edgecolor="k", lw=0.7)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(d_labels, fontsize=9)
    axes[1].set_ylabel("Cooling acc. (%)")
    axes[1].set_title("History dominance: does fatigue erase cooling?")
    axes[1].set_ylim(40, 105)
    drop_b = dom_results["drop_4b"]
    axes[1].annotate(f"drop={drop_b:+.1f}pp",
                     xy=(0.5, max(d_accs[0], d_accs[1]) - 5),
                     ha="center", fontsize=10,
                     color="red" if drop_b > 5 else "green")

    # Right: regression R²
    r2_labels = ["5A Fast", "5B Slow", "5C All"]
    r2_means  = [np.mean(reg_results["5A"]["r2s"]),
                 np.mean(reg_results["5B"]["r2s"]),
                 np.mean(reg_results["5C"]["r2s"])]
    r2_stds   = [np.std(reg_results["5A"]["r2s"]),
                 np.std(reg_results["5B"]["r2s"]),
                 np.std(reg_results["5C"]["r2s"])]
    axes[2].bar(range(3), r2_means, yerr=r2_stds, capsize=5,
                color=["#e6194b", "#4363d8", "#3cb44b"],
                alpha=0.85, edgecolor="k", lw=0.7)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(r2_labels)
    axes[2].set_ylabel("R²")
    axes[2].set_title("Fatigue regression R² by cooling type")
    axes[2].set_ylim(0, 1)

    plt.suptitle("Multi-History Glass: Experimental Summary", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary figure → {out_path}")


def plot_permutation_importance(perm: Dict, out_path: str) -> None:
    """Side-by-side bar charts of feature importance per task."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, task, color, title in [
            (axes[0], "cooling", "#e6194b", "6A Cooling classifier"),
            (axes[1], "fatigue", "#4363d8", "6B Fatigue classifier (3-class)")]:
        drops = perm[task]["drops"]
        order = np.argsort(drops)[::-1]
        names = [FEAT_NAMES[i] for i in order]
        vals  = [drops[i] for i in order]
        bar_colors = [color if v >= 0 else "#aaaaaa" for v in vals]
        ax.barh(range(8), vals,
                color=bar_colors,
                alpha=0.85, edgecolor="k", lw=0.5)
        ax.set_yticks(range(8))
        ax.set_yticklabels(names)
        ax.axvline(0, lw=0.8, c="k")
        ax.set_xlabel("AUC drop when feature zeroed")
        ax.set_title(f"{title}\n(intact AUC={perm[task]['auc_intact']:.4f})")
        ax.invert_yaxis()

    mean_diff = perm["mean_rank_diff"]
    plt.suptitle(
        f"Feature Importance per Task\n"
        f"Mean rank difference = {mean_diff:.2f}  "
        f"({'Different features' if mean_diff >= 2.5 else 'Similar features'})",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Permutation importance figure → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# 8.  FINAL RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────

def print_final_table(st: Dict, mt: Dict, dom: Dict,
                      reg: Dict, perm: Dict) -> None:
    """
    Print the publication-ready results table.
    This mirrors the format of Table IX in Paper 1.
    """
    print(f"\n{'='*72}")
    print(f"  COMPLETE RESULTS TABLE")
    print(f"{'─'*72}")
    print(f"  {'Experiment':<38}  {'Cool acc':>9}  {'Fat acc':>9}  "
          f"{'AUC':>7}")
    print(f"{'─'*72}")

    def _row(name, acc_c, acc_f, auc):
        acc_c_s = f"{acc_c:.1f}%" if acc_c > 0 else "    —"
        acc_f_s = f"{acc_f:.1f}%" if acc_f > 0 else "    —"
        auc_s   = f"{auc:.4f}"  if auc  > 0 else "    —"
        print(f"  {name:<38}  {acc_c_s:>9}  {acc_f_s:>9}  {auc_s:>7}")

    _row("1A Single-task cooling",
         np.mean(st["1A_cooling"]["accs"]),
         0, np.mean(st["1A_cooling"]["aucs"]))
    _row("1B Single-task fatigue (3-class)",
         0, np.mean(st["1B_fatigue"]["accs"]), 0)
    _row("2  Multi-task (cooling)",
         np.mean(mt["accs_c"]), 0, np.mean(mt["aucs_c"]))
    _row("2  Multi-task (fatigue)",
         0, np.mean(mt["accs_f"]), 0)
    _row("4A Cooling on pristine only",
         dom["acc_pristine"], 0, dom["auc_pristine"])
    _row("4B Cooling on fatigued (c400)",
         dom["acc_fat400"], 0, dom["auc_fat400"])

    print(f"{'─'*72}")
    print(f"  {'Orthogonality metric':<38}  {'Value':>9}")
    print(f"{'─'*72}")
    orth = mt["orthogonality"]
    print(f"  {'Distinct PCs for each task':<38}  "
          f"{'YES' if orth['orthogonal'] else 'NO':>9}")
    print(f"  {'Cross-contamination':<38}  "
          f"{orth['cross_contamination']:>9.4f}")
    print(f"  {'Cooling acc drop (c0→c400)':<38}  "
          f"{dom['drop_4b']:>+8.2f}%")
    print(f"  {'Mean feature rank diff (cool vs fat)':<38}  "
          f"{perm['mean_rank_diff']:>9.2f}")
    print(f"{'─'*72}")
    print(f"  {'Regression R²':<38}  {'Fast':>9}  {'Slow':>9}  {'All':>7}")
    print(f"{'─'*72}")
    print(f"  {'Fatigue regression (cycle number)':<38}  "
          f"{np.mean(reg['5A']['r2s']):>9.4f}  "
          f"{np.mean(reg['5B']['r2s']):>9.4f}  "
          f"{np.mean(reg['5C']['r2s']):>7.4f}")
    print(f"{'='*72}")

    # One-sentence interpretation
    print(f"\n  OVERALL INTERPRETATION")
    print(f"  {orth['interpretation']}")
    print(f"  {dom['interpretation']}")


# ──────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_global = time.time()
    _print_header()

    # ── Load data ──────────────────────────────────────────────────────────
    graphs = load_and_build_graphs(DATA_PATH)

    # ── Exp 1: Single-task baselines ───────────────────────────────────────
    st_results = exp_singletask(graphs)

    # ── Exp 2 + 3: Multi-task + orthogonality ─────────────────────────────
    mt_results = exp_multitask_and_orthogonality(graphs)

    # ── Exp 4: History dominance ───────────────────────────────────────────
    dom_results = exp_history_dominance(graphs)

    # ── Exp 5: Fatigue regression ──────────────────────────────────────────
    reg_results = exp_regression(graphs)

    # ── Exp 6: Permutation importance ─────────────────────────────────────
    perm_results = exp_permutation_importance(graphs)

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n  Generating figures...")
    plot_orthogonality(
        mt_results["orthogonality"],
        os.path.join(OUT_DIR, "fig_orthogonality.png"))
    plot_results_summary(
        st_results, mt_results, dom_results, reg_results,
        os.path.join(OUT_DIR, "fig_summary.png"))
    plot_permutation_importance(
        perm_results,
        os.path.join(OUT_DIR, "fig_permutation_importance.png"))

    # ── Final table ────────────────────────────────────────────────────────
    print_final_table(st_results, mt_results, dom_results,
                      reg_results, perm_results)

    # ── Save all results ───────────────────────────────────────────────────
    save_path = os.path.join(OUT_DIR, "all_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "single_task":   st_results,
            "multi_task":    mt_results,
            "dominance":     dom_results,
            "regression":    reg_results,
            "permutation":   perm_results,
        }, f)
    print(f"\n  All results saved → {save_path}")

    total_min = (time.time() - t_global) / 60
    print(f"  Total runtime: {total_min:.1f} min\n")


if __name__ == "__main__":
    # Required on Windows to prevent DataLoader worker processes from
    # spawning recursively.
    import multiprocessing
    multiprocessing.freeze_support()
    main()