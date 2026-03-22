#!/usr/bin/env python3
"""
==========================================================================
DENSE EARLY-CYCLE DATA GENERATION  —  Arrhenius Forgetting Experiment
Paper: "Geometric Orthogonality of Competing Structural Memories in Glass"
==========================================================================

PURPOSE
────────
The original 9-cycle datagen used SAVE_AT = [0,50,100,...,400].
At T ≥ 0.38, the forgetting curve collapses within the first 50 cycles
(τ < 50), making the timescale unresolvable with 50-cycle intervals.

This script uses dense early sampling:
    SAVE_AT = [0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

MAX_CYCLES is reduced to 200 (the curve saturates by cycle 100–150 at
all temperatures ≥ 0.38 Tg, saving ~50% of simulation time).

WHICH TEMPERATURES TO REGENERATE
──────────────────────────────────
  T=0.35 (0.78Tg): τ≈79 cycles — ALREADY RESOLVED, do NOT regenerate
  T=0.38 (0.84Tg): τ<50 cycles — regenerate  ← run this script
  T=0.40 (0.89Tg): τ<50 cycles — regenerate
  T=0.42 (0.93Tg): τ<50 cycles — regenerate
  T=0.44 (0.98Tg): τ<50 cycles — regenerate
  T=0.46 (1.02Tg): τ<50 cycles — regenerate

DUAL-GPU LAUNCHER (Kaggle)
───────────────────────────
Running without --gpu_id spawns one worker per temperature on each GPU.
Edit _TB_LIST to match which temperatures you want to generate in parallel.

Example — run T=0.38 and T=0.40 in parallel on 2 Kaggle T4s:
    _TB_LIST = [0.38, 0.40]

Then rerun with:
    _TB_LIST = [0.42, 0.44]
And again with:
    _TB_LIST = [0.46]

USAGE
──────
  # Write to Kaggle working dir first (in a cell):
  %%writefile /kaggle/working/datagen_dense.py
  <paste this file>

  # Then run:
  !python -u /kaggle/working/datagen_dense.py

  # Local single-temperature run:
  python datagen_dense.py --gpu_id 0 --t_battery 0.38

OUTPUT FILES
─────────────
  multihistory_dense_{T_tag}.pkl
  e.g. multihistory_dense_t38.pkl, multihistory_dense_t40.pkl, ...

  These are SEPARATE from the original 9-cycle files.
  The Arrhenius analysis script will load them by this naming convention.

ESTIMATED RUNTIME (Kaggle T4)
──────────────────────────────
  ~20–25 min per temperature
  2 temperatures in parallel on dual-GPU = ~25 min per pair
  5 temperatures total = ~65 min wall-clock (3 launcher runs)

  Local RTX 4060: ~18–22 min per temperature
==========================================================================
"""

# ══════════════════════════════════════════════════════════════════════════
#  SECTION 0 — LAUNCHER
# ══════════════════════════════════════════════════════════════════════════
import argparse as _ap
import os        as _os
import subprocess as _sub
import sys        as _sys
import threading  as _th
import time       as _time

_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--gpu_id",    type=int,   default=None)
_parser.add_argument("--t_battery", type=float, default=None)
_args, _ = _parser.parse_known_args()

if _args.gpu_id is None:

    _SCRIPT  = _os.path.abspath(__file__)

    # ── EDIT THIS to choose which pair to run ────────────────────────────
    # Run 1: [0.38, 0.40]
    # Run 2: [0.42, 0.44]
    # Run 3: [0.46]          (single GPU, GPU 0 only)
    _TB_LIST = [0.38, 0.40]
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("  DENSE EARLY-CYCLE LAUNCHER")
    for i, tb in enumerate(_TB_LIST):
        print(f"    GPU {i}  →  T_BATTERY = {tb}")
    print("="*60 + "\n", flush=True)

    def _stream(proc, prefix):
        for raw in iter(proc.stdout.readline, b""):
            print(f"{prefix} {raw.decode('utf-8', errors='replace').rstrip()}",
                  flush=True)

    procs, threads = [], []
    t0 = _time.time()

    for gpu_id, t_battery in enumerate(_TB_LIST):
        env = _os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]           = str(gpu_id)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

        cmd = [_sys.executable, "-u", _SCRIPT,
               f"--gpu_id={gpu_id}", f"--t_battery={t_battery}"]
        p = _sub.Popen(cmd, env=env, stdout=_sub.PIPE, stderr=_sub.STDOUT)
        procs.append((p, gpu_id, t_battery))

        t = _th.Thread(target=_stream,
                       args=(p, f"[G{gpu_id}|T{t_battery}]"), daemon=True)
        t.start()
        threads.append(t)
        print(f"  Launched PID {p.pid}  GPU {gpu_id}  T={t_battery}",
              flush=True)

    for p, gpu_id, t_battery in procs:
        p.wait()
        rc  = p.returncode
        print(f"\n  GPU {gpu_id}  T={t_battery}: "
              f"{'DONE ✓' if rc == 0 else f'FAILED rc={rc}'}", flush=True)

    for t in threads:
        t.join()

    print(f"\n  Wall-clock: {(_time.time()-t0)/60:.1f} min")
    print("  Files in: /kaggle/working/multihistory_data/ (or ./multihistory_data/)")
    _sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — WORKER
# ══════════════════════════════════════════════════════════════════════════

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",  "false")
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import os, sys, time, pickle, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random
import jax.lax as lax

GPU_ID    = _args.gpu_id
T_BATTERY = _args.t_battery

print(f"JAX     : {jax.__version__}")
print(f"Devices : {jax.devices()}")
print(f"GPU_ID  : {GPU_ID}   T_BATTERY : {T_BATTERY}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

N_ATOMS   = 256
RHO       = 1.2
BOX_L     = float((N_ATOMS / RHO) ** (1.0 / 3.0))   # ≈ 5.975 σ
DT        = 5e-4
RC_LJ     = 2.5
RC_GRAPH  = 1.5
SCAN_CHUNK = 100

T_HIGH = 2.0
T_LOW  = 0.10
TG     = 0.45

FAST_COOL_CHUNKS  = 40
SLOW_COOL_CHUNKS  = 4000
HOT_EQUIL_CHUNKS  = 10
FAST_FINAL_CHUNKS = 20
SLOW_FINAL_CHUNKS = 200

STRAIN_AMP  = 0.08
STEPS_PHASE = 500

# ── KEY CHANGE: dense early sampling, MAX_CYCLES halved ──────────────────
MAX_CYCLES = 200
SAVE_AT    = [0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

N_GLASSES   = 100
MAX_RETRIES = 5

# Output naming: multihistory_dense_{t_tag}.pkl
# e.g. T=0.38 → multihistory_dense_t38.pkl
_t_tag  = str(T_BATTERY).replace("0.", "").replace(".", "")  # "038", "040" etc
OUT_DIR  = "multihistory_data"
OUT_FILE = os.path.join(OUT_DIR, f"multihistory_dense_t{_t_tag}.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

# Physical validity thresholds (same as original)
HARD_CORE_MIN      = 0.75
COORD_MIN          = 6;  COORD_MAX = 18
MIN_COORD_PER_ATOM = 1
BOND_MEAN_LO       = 0.95; BOND_MEAN_HI = 1.40
CRYSTAL_STD_MAX    = 0.05
BOND_STD_LO        = 0.05; BOND_STD_HI = 0.35
ENERGY_LO          = -10.0; ENERGY_HI = -2.0
MONOTONE_TOL       = 0.025
MEAN_DRIFT_MAX     = 0.08


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — JAX PHYSICS  (identical to original)
# ══════════════════════════════════════════════════════════════════════════

@jit
def lj_forces_pbc(pos: jnp.ndarray, box: float) -> jnp.ndarray:
    dr    = pos[:, None, :] - pos[None, :, :]
    dr    = dr - box * jnp.round(dr / box)
    r2    = jnp.sum(dr ** 2, axis=-1)
    mask  = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)
    r2s   = jnp.where(mask, r2, 1.0)
    inv2  = 1.0 / r2s
    inv6  = inv2 ** 3
    inv12 = inv6 ** 2
    coeff = jnp.where(mask, 24.0 * inv2 * (2.0 * inv12 - inv6), 0.0)
    return jnp.sum(coeff[:, :, None] * dr, axis=1)


@jit
def lj_energy_pbc(pos: jnp.ndarray, box: float) -> float:
    dr     = pos[:, None, :] - pos[None, :, :]
    dr     = dr - box * jnp.round(dr / box)
    r2     = jnp.sum(dr ** 2, axis=-1)
    mask   = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)
    r2s    = jnp.where(mask, r2, 1.0)
    inv6   = (1.0 / r2s) ** 3
    pair_e = jnp.where(mask, 4.0 * (inv6 ** 2 - inv6), 0.0)
    return 0.5 * jnp.sum(pair_e) / pos.shape[0]


@jit
def md_chunk(pos: jnp.ndarray, key: jnp.ndarray,
             box: float, temp: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dt        = jnp.float32(DT)
    sqrt_2Tdt = jnp.sqrt(2.0 * temp * dt)

    def _step(carry, _):
        p, k  = carry
        f     = lj_forces_pbc(p, box)
        f     = jnp.clip(f, -50.0, 50.0)
        k, sk = random.split(k)
        noise = random.normal(sk, p.shape, dtype=jnp.float32)
        p_new = p + f * dt + sqrt_2Tdt * noise
        return (p_new % box, k), None

    (pos_out, key_out), _ = lax.scan(_step, (pos, key), None,
                                     length=SCAN_CHUNK)
    return pos_out, key_out


def run_md(pos, key, box, temp, n_steps):
    n_calls = max(1, int(np.ceil(n_steps / SCAN_CHUNK)))
    pos_j   = jnp.array(pos, dtype=jnp.float32)
    for _ in range(n_calls):
        pos_j, key = md_chunk(pos_j, key,
                               jnp.float32(box), jnp.float32(temp))
    return np.array(pos_j, dtype=np.float32), key


def warmup():
    print("Warming up JAX …", end="", flush=True)
    t0 = time.time()
    _p = jnp.zeros((N_ATOMS, 3), dtype=jnp.float32)
    _k = random.PRNGKey(0)
    _p2, _ = md_chunk(_p, _k, jnp.float32(BOX_L), jnp.float32(T_LOW))
    _e = lj_energy_pbc(_p2, jnp.float32(BOX_L))
    jax.block_until_ready(_p2); jax.block_until_ready(_e)
    print(f"  {time.time()-t0:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — VALIDATION  (same logic as original, condensed)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class VR:
    passed:   bool = True
    metrics:  Dict[str, float] = field(default_factory=dict)
    errors:   List[str]        = field(default_factory=list)
    warnings: List[str]        = field(default_factory=list)

    def check(self, name, ok, value=None, fmt=".4f", msg=""):
        if value is not None:
            self.metrics[name] = float(value)
        if not ok:
            self.passed = False
            s = f"FAIL [{name}]"
            if value is not None: s += f" val={value:{fmt}}"
            if msg: s += f" — {msg}"
            self.errors.append(s)


def validate(pos, box):
    v = VR()
    if not np.isfinite(pos).all():
        v.check("C1_nan", False, msg="NaN/Inf"); return v
    v.check("C2_box", bool(np.all(pos >= 0) and np.all(pos < box)))

    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box * np.round(dr / box)
    dist = np.sqrt(np.sum(dr**2, axis=-1))
    nbr  = (dist > 1e-6) & (dist < RC_GRAPH)

    upper = np.triu_indices(len(pos), k=1)
    min_d = float(dist[upper].min())
    v.check("C3_hardcore", min_d >= HARD_CORE_MIN, min_d)

    coord  = nbr.sum(axis=1).astype(float)
    mc     = float(coord.mean())
    v.check("C4_coord",    COORD_MIN <= mc <= COORD_MAX, mc, ".2f")
    v.check("C5_isolated", int((coord < 1).sum()) == 0)

    bl = dist[nbr]
    if len(bl) == 0:
        for c in ("C6","C7","C9"): v.check(c, False, msg="no bonds")
        return v

    mean_bl = float(bl.mean()); std_bl = float(bl.std())
    v.check("C6_bondmean", BOND_MEAN_LO <= mean_bl <= BOND_MEAN_HI, mean_bl)
    v.check("C7_bondstd",  BOND_STD_LO  <= std_bl  <= BOND_STD_HI,  std_bl)
    try:
        e = float(lj_energy_pbc(jnp.array(pos, dtype=jnp.float32),
                                 jnp.float32(box)))
        v.check("C8_energy", ENERGY_LO <= e <= ENERGY_HI, e)
    except Exception as ex:
        v.check("C8_energy", False, msg=str(ex)); e = np.nan
    v.check("C9_crystal",  std_bl > CRYSTAL_STD_MAX, std_bl)
    v.check("C10_hetero",  float(coord.std()) > 0)

    v.metrics.update({
        "mean_bond_length": mean_bl, "std_bond_length": std_bl,
        "mean_coord": mc,            "energy_per_atom": e,
        "min_pair_dist": min_d,
    })
    return v


def check_fatigue(bond_stds, mean_bonds):
    """Soft check — informational only, does not gate inclusion."""
    cyc = sorted(bond_stds)
    if len(cyc) < 2:
        return True, ""
    c0, cN = cyc[0], cyc[-1]
    net = bond_stds[cN] - bond_stds[c0]
    drift = abs(mean_bonds[cN] - mean_bonds[c0])
    ok  = (net >= -MONOTONE_TOL/2) and (drift < MEAN_DRIFT_MAX)
    msg = f"net_std_change={net:+.5f}  drift={drift:.5f}"
    return ok, msg


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — GLASS GENERATION  (identical to original)
# ══════════════════════════════════════════════════════════════════════════

def init_lattice(key):
    n = int(np.ceil(N_ATOMS ** (1/3)))
    sp = BOX_L / n
    pts = np.array([(i,j,k) for i in range(n)
                             for j in range(n)
                             for k in range(n)],
                   dtype=np.float32)[:N_ATOMS] * sp
    key, sk = random.split(key)
    noise = np.array(random.normal(sk, pts.shape, dtype=jnp.float32)) * 0.05
    return ((pts + noise) % BOX_L).astype(np.float32), key


def generate_glass(key, n_cool, n_final, label=""):
    box = jnp.float32(BOX_L)
    pos, key = init_lattice(key)
    pj = jnp.array(pos, dtype=jnp.float32)
    for _ in range(HOT_EQUIL_CHUNKS):
        pj, key = md_chunk(pj, key, box, jnp.float32(T_HIGH))
    for T in np.linspace(T_HIGH, T_LOW, n_cool, dtype=np.float32):
        pj, key = md_chunk(pj, key, box, jnp.float32(T))
    for _ in range(n_final):
        pj, key = md_chunk(pj, key, box, jnp.float32(T_LOW))
    pos_out = np.array(pj, dtype=np.float32)
    return pos_out, key, validate(pos_out, BOX_L)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FATIGUE PROTOCOL
# ══════════════════════════════════════════════════════════════════════════

def one_cycle(pos, key):
    scale = 1.0 + STRAIN_AMP
    Lexp  = BOX_L * scale
    pj = jnp.array(pos) * jnp.float32(scale)
    pj, key = run_md(pj, key, Lexp,  T_BATTERY, STEPS_PHASE)
    pj = pj / jnp.float32(scale)
    pj, key = run_md(pj, key, BOX_L, T_BATTERY, STEPS_PHASE)
    return np.array(pj, dtype=np.float32), key


def make_sample(pos, glass_id, cooling_type, fatigue_cycles, vr):
    fat_label = SAVE_AT.index(fatigue_cycles) if fatigue_cycles in SAVE_AT else -1
    return {
        "glass_id":            glass_id,
        "cooling_type":        cooling_type,
        "cooling_chunks":      FAST_COOL_CHUNKS if cooling_type=="fast" else SLOW_COOL_CHUNKS,
        "fatigue_cycles":      fatigue_cycles,
        "t_battery":           T_BATTERY,
        "label_cooling":       0 if cooling_type == "fast" else 1,
        "label_fatigue":       fat_label,
        "label_fatigue_cont":  fatigue_cycles / MAX_CYCLES,
        "positions":           pos.astype(np.float32),
        "validation":          vr.metrics,
        "sample_passed":       vr.passed,
        "validation_errors":   vr.errors,
        "validation_warnings": vr.warnings,
    }


def run_fatigue(pos0, key, glass_id, cooling_type):
    pos  = pos0.copy()
    snaps = []

    vr0 = validate(pos, BOX_L)
    if 0 in SAVE_AT:
        snaps.append(make_sample(pos, glass_id, cooling_type, 0, vr0))

    for cyc in range(1, MAX_CYCLES + 1):
        pos, key = one_cycle(pos, key)
        if cyc in SAVE_AT:
            vrc = validate(pos, BOX_L)
            snaps.append(make_sample(pos, glass_id, cooling_type, cyc, vrc))

    # Soft fatigue progression check (informational)
    bstds  = {s["fatigue_cycles"]: s["validation"].get("std_bond_length", np.nan)
              for s in snaps if "std_bond_length" in s["validation"]}
    mbonds = {s["fatigue_cycles"]: s["validation"].get("mean_bond_length", np.nan)
              for s in snaps if "mean_bond_length" in s["validation"]}
    fat_ok, fat_msg = check_fatigue(bstds, mbonds)
    if not fat_ok:
        print(f"    FATIGUE WARN glass {glass_id}: {fat_msg}", flush=True)

    return snaps


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def generate():
    dataset    = []
    master_key = random.PRNGKey(42 + (GPU_ID or 0))
    t0         = time.time()
    n_failed   = 0

    target = 2 * N_GLASSES * len(SAVE_AT)
    print(f"\n{'='*56}")
    print(f"  T={T_BATTERY} ({T_BATTERY/TG:.2f} Tg)")
    print(f"  SAVE_AT = {SAVE_AT}")
    print(f"  MAX_CYCLES = {MAX_CYCLES}  (dense early sampling)")
    print(f"  Target: {target} samples")
    print(f"  Output: {OUT_FILE}")
    print(f"{'='*56}\n", flush=True)

    for cooling_type, n_cool, n_final in [
            ("fast", FAST_COOL_CHUNKS, FAST_FINAL_CHUNKS),
            ("slow", SLOW_COOL_CHUNKS, SLOW_FINAL_CHUNKS),
    ]:
        print(f"\n── {cooling_type.upper()}  ({n_cool} chunks) ──", flush=True)
        t_ct = time.time()
        n_ok = 0

        for gid in range(N_GLASSES):
            pos = None
            for attempt in range(MAX_RETRIES):
                master_key, sk = random.split(master_key)
                try:
                    p, master_key, vr = generate_glass(sk, n_cool, n_final,
                                                        label=cooling_type)
                except Exception as ex:
                    print(f"  glass {gid} attempt {attempt+1}: {ex}", flush=True)
                    continue
                if vr.passed:
                    pos = p; break
                elif attempt == MAX_RETRIES - 1:
                    print(f"  WARNING: glass {gid} all retries failed: "
                          + " | ".join(vr.errors), flush=True)
                    n_failed += 1

            if pos is None:
                continue

            master_key, ck = random.split(master_key)
            try:
                snaps = run_fatigue(pos, ck, gid, cooling_type)
            except Exception as ex:
                print(f"  fatigue failed glass {gid}: {ex}", flush=True)
                continue

            valid = [s for s in snaps if s["sample_passed"]]
            n_bad = len(snaps) - len(valid)
            if n_bad:
                print(f"  glass {gid}: {n_bad} snapshots dropped", flush=True)

            dataset.extend(valid)
            n_ok += 1

            if (gid + 1) % 10 == 0:
                el  = time.time() - t_ct
                eta = el / (gid+1) * (N_GLASSES - gid - 1)
                print(f"  [{gid+1:3d}/{N_GLASSES}]  "
                      f"{el/60:.1f}min | ETA {eta/60:.1f}min | "
                      f"samples={len(dataset)}", flush=True)

        print(f"  {cooling_type.upper()} done: {n_ok}/{N_GLASSES}  "
              f"{(time.time()-t_ct)/60:.1f}min", flush=True)

    total = time.time() - t0
    print(f"\n  COMPLETE: {len(dataset)} samples  {total/60:.1f}min  "
          f"failures={n_failed}", flush=True)
    return dataset


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════════

def report(dataset):
    from collections import defaultdict
    cells = defaultdict(list)
    for s in dataset:
        cells[(s["cooling_type"], s["fatigue_cycles"])].append(s)

    print(f"\n{'='*72}")
    print(f"  VALIDATION REPORT  T={T_BATTERY}")
    print(f"{'─'*72}")
    print(f"  {'Cool':<5} {'Cyc':>4}  {'N':>4}  {'Pass%':>5}  "
          f"{'MeanBond':>9}  {'StdBond':>9}  {'E/N':>8}")
    print(f"{'─'*72}")

    for (ct, cyc), sl in sorted(cells.items()):
        vms   = [s["validation"] for s in sl if s["validation"]]
        npass = sum(1 for s in sl if s["sample_passed"])

        def _m(k):
            v = [x.get(k, np.nan) for x in vms
                 if np.isfinite(x.get(k, np.nan))]
            return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)

        mb = _m("mean_bond_length"); sb = _m("std_bond_length")
        en = _m("energy_per_atom")
        print(f"  {ct:<5} {cyc:>4}  {len(sl):>4}  "
              f"{100*npass/max(len(sl),1):>4.0f}%  "
              f"{mb[0]:>7.4f}±{mb[1]:.4f}  "
              f"{sb[0]:>7.5f}±{sb[1]:.5f}  "
              f"{en[0]:>6.3f}±{en[1]:.3f}")

    print(f"{'='*72}\n", flush=True)

    # Conditioning dip check (fast vs slow at early cycles)
    print("  CONDITIONING DIP CHECK (fast glasses, bond-std vs cycle):")
    fast_snaps = {s["fatigue_cycles"]: [] for s in dataset if s["cooling_type"]=="fast"}
    for s in dataset:
        if s["cooling_type"] == "fast":
            fast_snaps[s["fatigue_cycles"]].append(
                s["validation"].get("std_bond_length", np.nan))
    for cyc in SAVE_AT[:8]:   # show first 8 cycle points
        vals = [v for v in fast_snaps.get(cyc, []) if np.isfinite(v)]
        if vals:
            print(f"    cycle {cyc:3d}: σ_r = {np.mean(vals):.5f} ± {np.std(vals):.5f}")
    print(flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SAVE
# ══════════════════════════════════════════════════════════════════════════

def save(dataset, path):
    with open(path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(path) / 1e6
    print(f"  Saved → {path}  ({mb:.1f} MB)")

    csv = path.replace(".pkl", "_summary.csv")
    with open(csv, "w") as f:
        f.write("idx,glass_id,cooling_type,fatigue_cycles,t_battery,"
                "label_cooling,label_fatigue,label_fatigue_cont,"
                "sample_passed,mean_bond,std_bond,energy\n")
        for i, s in enumerate(dataset):
            vm = s.get("validation", {})
            f.write(f"{i},{s['glass_id']},{s['cooling_type']},"
                    f"{s['fatigue_cycles']},{s['t_battery']},"
                    f"{s['label_cooling']},{s['label_fatigue']},"
                    f"{s['label_fatigue_cont']:.4f},{int(s['sample_passed'])},"
                    f"{vm.get('mean_bond_length',float('nan')):.5f},"
                    f"{vm.get('std_bond_length', float('nan')):.5f},"
                    f"{vm.get('energy_per_atom', float('nan')):.4f}\n")
    print(f"  CSV  → {csv}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def main():
    if os.path.exists(OUT_FILE):
        print(f"\n  {OUT_FILE} already exists — loading for report.",
              flush=True)
        with open(OUT_FILE, "rb") as f:
            dataset = pickle.load(f)
        report(dataset)
        return

    warmup()
    dataset = generate()
    report(dataset)
    save(dataset, OUT_FILE)

    n_pass = sum(1 for s in dataset if s["sample_passed"])
    print(f"\n  Pass rate: {n_pass}/{len(dataset)} "
          f"({100*n_pass/max(len(dataset),1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
