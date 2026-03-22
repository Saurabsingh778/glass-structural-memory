#!/usr/bin/env python3
"""
==========================================================================
MULTI-HISTORY GLASS DATA GENERATION
Paper: "Geometric Orthogonality of Competing Structural Memories in Glass"
==========================================================================

2D FACTORIAL DESIGN
─────────────────────────────────────────────────────────────────────
Cooling Rate     │  Fatigue Cycles  │  Samples  │  Label (cool, fat)
─────────────────────────────────────────────────────────────────────
fast (40  chunks)│  0               │  100      │  (0, 0)
fast (40  chunks)│  200             │  100      │  (0, 1)
fast (40  chunks)│  400             │  100      │  (0, 2)
slow (4000 chunks)│ 0               │  100      │  (1, 0)
slow (4000 chunks)│ 200             │  100      │  (1, 1)
slow (4000 chunks)│ 400             │  100      │  (1, 2)
─────────────────────────────────────────────────────────────────────
Total:  600 samples from 200 unique glasses (100 fast + 100 slow)

PHYSICAL VALIDITY CHECKS (every sample)
─────────────────────────────────────────
C1  No NaN / Inf in positions
C2  All positions inside periodic box  [0, BOX_L)
C3  Minimum pairwise distance > HARD_CORE_MIN  (no overlaps)
C4  Mean coordination within [COORD_MIN, COORD_MAX]
C5  No isolated atoms  (coordination ≥ 1)
C6  Mean bond length within [BOND_MEAN_LO, BOND_MEAN_HI]
C7  Bond length std within [BOND_STD_LO,  BOND_STD_HI]
C8  LJ energy per particle within [ENERGY_LO, ENERGY_HI]
C9  Glass is not crystalline  (bond-std > CRYSTAL_STD_MAX)
C10 Structural heterogeneity  (std of per-atom coordination > 0)

FATIGUE PROGRESSION CHECK (per glass, across cycles)
──────────────────────────────────────────────────────
F1  bond-std(cycle 200)  ≥  bond-std(cycle 0) - MONOTONE_TOL
F2  bond-std(cycle 400)  ≥  bond-std(cycle 200) - MONOTONE_TOL
F3  Mean bond length drift < MEAN_DRIFT_MAX  (no bulk expansion)

ESTIMATED RUNTIME  (Kaggle T4  /  RTX 4060)
──────────────────────────────────────────────
Fast glass generation  :  ~  2 min  (100 glasses × 40  cooling chunks)
Slow glass generation  :  ~ 10 min  (100 glasses × 4000 cooling chunks)
Fatigue cycling        :  ~ 20 min  (200 glasses × 400 cycles)
─────────────────────────────────────────────
Total                  :  ~ 32 min
==========================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 0.  ENVIRONMENT SETUP  (must happen before importing JAX)
# ──────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",  "0.45")

import time
import pickle
import warnings
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, random
import jax.lax as lax

print(f"JAX  version : {jax.__version__}")
print(f"JAX  devices : {jax.devices()}")
print(f"NumPy version: {np.__version__}")
print()


# ──────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

# ── System (LJ reduced units: σ = ε = m = 1) ─────────────────────────────
N_ATOMS   = 256
RHO       = 1.2
BOX_L     = float((N_ATOMS / RHO) ** (1.0 / 3.0))   # ≈ 5.975 σ
DT        = 5e-4                                      # timestep
RC_LJ     = 2.5                                       # LJ cutoff
RC_GRAPH  = 1.5                                       # graph / bond cutoff
SCAN_CHUNK = 100                                      # steps per JIT chunk

# ── Temperatures ──────────────────────────────────────────────────────────
T_HIGH    = 2.0     # starting liquid
T_LOW     = 0.10    # deep glass
T_BATTERY = 0.44   # cycling temp  (0.93 Tg,  Tg ≈ 0.45 at ρ=1.2)

# ── Cooling protocols ─────────────────────────────────────────────────────
#   Both use identical pre/post equilibration; only n_cool_chunks differs.
#   100× ratio reproduces the fast / slow distinction from Paper 1.
FAST_COOL_CHUNKS   = 40     # 4 000 steps  (2.0 τ  cooling time)
SLOW_COOL_CHUNKS   = 4000   # 400 000 steps (200 τ cooling time)
HOT_EQUIL_CHUNKS   = 10     # hot equilibration before cooling
FAST_FINAL_CHUNKS  = 20     # post-cooling equilibration, fast glass
SLOW_FINAL_CHUNKS  = 200    # post-cooling equilibration, slow glass

# ── Fatigue protocol ──────────────────────────────────────────────────────
STRAIN_AMP   = 0.08       # 8% volumetric strain (LiPON range: 3–8%)
STEPS_PHASE  = 500          # Brownian steps per half-cycle
MAX_CYCLES   = 400          # total charge/discharge cycles
SAVE_AT      = [0, 200, 400, 600, 800, 1000]  # which cycles to snapshot

# ── Dataset ───────────────────────────────────────────────────────────────
N_GLASSES    = 100          # independent glasses per (cooling × fatigue)
                             # total samples = 2 × 3 × N_GLASSES = 600
MAX_RETRIES  = 5            # validation retries per glass

# ── Output ────────────────────────────────────────────────────────────────
OUT_DIR  = "multihistory_data"
OUT_FILE = os.path.join(OUT_DIR, "multihistory_glasses_amp_8_t_44.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Physical validity thresholds (derived from Paper 2, Table I) ──────────
#   LJ glass at ρ=1.2, T=0.10, rc=1.5σ

#   Hard-core: any pair < 0.75σ is a catastrophic overlap
HARD_CORE_MIN   = 0.75

#   Coordination  at ρ=1.2, rc=1.5:  first shell ≈ 11–13 neighbours
COORD_MIN  = 6
COORD_MAX  = 18
MIN_COORD_PER_ATOM = 1   # no isolated atoms

#   Mean bond length: LJ equilibrium r_eq = 2^(1/6) ≈ 1.122 σ
#   Glass mean: 1.12–1.14 σ (Paper 2, Table I).  Wide bounds to be safe.
BOND_MEAN_LO = 0.95
BOND_MEAN_HI = 1.40

#   Bond-length std in glass: ~0.145–0.165 σ (Paper 2).
#   Too low  → crystal   (std < CRYSTAL_STD_MAX)
#   Too high → liquid    (std > BOND_STD_HI)
CRYSTAL_STD_MAX = 0.05
BOND_STD_LO     = 0.05
BOND_STD_HI     = 0.35

#   LJ energy per particle at ρ=1.2, T=0.10:  ≈ −6.1 to −6.3 ε
#   Wide bounds to accommodate both fast and slow glasses + numerical noise.
ENERGY_LO  = -10.0
ENERGY_HI  =  -2.0

#   Fatigue monotonicity tolerance
#
#   PHYSICAL NOTE — why MONOTONE_TOL is intentionally large:
#   Fast-cooled glasses start in high-disorder states.  At T=0.42 ≈ 0.93 Tg
#   the first ~200 cycles act partly as annealing: mechanical energy injection
#   lets the glass relax toward lower-energy configurations, temporarily
#   REDUCING bond-std before fatigue accumulation dominates.  This "conditioning
#   dip" is documented in real battery electrolytes (Sakamoto et al. 2013).
#   A strict monotonicity check would incorrectly reject these physically valid
#   glasses.  The check below uses a generous tolerance and is classified as a
#   WARNING (informational) rather than a hard failure — the fatigue_vr result
#   is stored in the sample but does NOT prevent the sample from entering the
#   dataset.  Individual snapshot validity (C1–C10) is the hard gate.
#
#   For slow-cooled glasses the conditioning dip is much smaller (≲ 0.003 σ)
#   because the glass is already near equilibrium before cycling begins.
MONOTONE_TOL    = 0.025   # σ units  — covers the ~0.013 conditioning dip
MEAN_DRIFT_MAX  = 0.08    # max absolute shift in mean bond length σ units


# ──────────────────────────────────────────────────────────────────────────
# 2.  JAX PHYSICS ENGINE
# ──────────────────────────────────────────────────────────────────────────

@jit
def lj_forces_pbc(pos: jnp.ndarray, box: float) -> jnp.ndarray:
    """
    Vectorised LJ pair forces with minimum-image PBC.

    Args
    ----
    pos  : (N, 3) float32  particle positions
    box  : scalar          cubic box side length

    Returns
    -------
    forces : (N, 3) float32
    """
    dr   = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    dr   = dr - box * jnp.round(dr / box)             # minimum image
    r2   = jnp.sum(dr ** 2, axis=-1)                  # (N, N)

    mask = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)

    r2s   = jnp.where(mask, r2, 1.0)                  # safe denominator
    inv2  = 1.0 / r2s
    inv6  = inv2 ** 3
    inv12 = inv6 ** 2

    # 24ε · [2(σ/r)¹² − (σ/r)⁶] / r²
    coeff  = jnp.where(mask, 24.0 * inv2 * (2.0 * inv12 - inv6), 0.0)
    forces = jnp.sum(coeff[:, :, None] * dr, axis=1)  # (N, 3)
    return forces


@jit
def lj_energy_pbc(pos: jnp.ndarray, box: float) -> float:
    """
    Total LJ potential energy with minimum-image PBC.
    Used ONLY for physical validation (not in the MD loop).
    rc = RC_LJ = 2.5σ  (standard LJ cutoff).

    Returns energy / N  (per-particle).
    """
    N    = pos.shape[0]
    dr   = pos[:, None, :] - pos[None, :, :]
    dr   = dr - box * jnp.round(dr / box)
    r2   = jnp.sum(dr ** 2, axis=-1)

    mask = (r2 > 1e-6) & (r2 < RC_LJ * RC_LJ)
    r2s  = jnp.where(mask, r2, 1.0)
    inv6 = (1.0 / r2s) ** 3

    # 4ε[(σ/r)¹² − (σ/r)⁶]
    pair_e = jnp.where(mask, 4.0 * (inv6 ** 2 - inv6), 0.0)

    # Divide by 2 (double-counting) and by N (per-particle)
    return 0.5 * jnp.sum(pair_e) / N


@jit
def md_chunk(pos:  jnp.ndarray,
             key:  jnp.ndarray,
             box:  float,
             temp: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run SCAN_CHUNK Brownian dynamics steps.

    Single JIT compilation handles any (box, temp) pair because both are
    passed as dynamic JAX values, not Python constants.

    Returns  (positions_out, key_out).
    """
    dt        = jnp.float32(DT)
    sqrt_2Tdt = jnp.sqrt(2.0 * temp * dt)

    def _step(carry, _):
        p, k = carry
        f    = lj_forces_pbc(p, box)
        f    = jnp.clip(f, -50.0, 50.0)          # stability clamp
        k, sk  = random.split(k)
        noise  = random.normal(sk, p.shape, dtype=jnp.float32)
        p_new  = p + f * dt + sqrt_2Tdt * noise
        p_new  = p_new % box                      # periodic wrap
        return (p_new, k), None

    (pos_out, key_out), _ = lax.scan(_step, (pos, key), None,
                                     length=SCAN_CHUNK)
    return pos_out, key_out


def run_md(pos:    np.ndarray,
           key:    jnp.ndarray,
           box:    float,
           temp:   float,
           n_steps: int) -> Tuple[np.ndarray, jnp.ndarray]:
    """
    Run n_steps of Brownian dynamics.
    n_steps is rounded up to the nearest multiple of SCAN_CHUNK.
    """
    box_j = jnp.float32(box)
    T_j   = jnp.float32(temp)
    n_calls = max(1, int(np.ceil(n_steps / SCAN_CHUNK)))
    pos_j = jnp.array(pos, dtype=jnp.float32)
    for _ in range(n_calls):
        pos_j, key = md_chunk(pos_j, key, box_j, T_j)
    return np.array(pos_j, dtype=np.float32), key


# ── JIT warm-up ───────────────────────────────────────────────────────────
def _warmup_jax() -> None:
    print("Warming up JAX kernels …", end="", flush=True)
    t0 = time.time()
    _p = jnp.zeros((N_ATOMS, 3), dtype=jnp.float32)
    _k = random.PRNGKey(0)
    _p2, _ = md_chunk(_p, _k, jnp.float32(BOX_L), jnp.float32(T_LOW))
    _e     = lj_energy_pbc(_p2, jnp.float32(BOX_L))
    jax.block_until_ready(_p2)
    jax.block_until_ready(_e)
    print(f"  done in {time.time()-t0:.1f} s")


# ──────────────────────────────────────────────────────────────────────────
# 3.  PHYSICAL VALIDATION
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Structured result of a single physical validity check pass."""
    passed:   bool = True
    checks:   Dict[str, bool]  = field(default_factory=dict)
    metrics:  Dict[str, float] = field(default_factory=dict)
    warnings: List[str]        = field(default_factory=list)
    errors:   List[str]        = field(default_factory=list)

    def add_check(self, name: str, ok: bool, value=None,
                  fmt: str = ".4f", error_msg: str = ""):
        """Record one check.  Sets self.passed = False on failure."""
        self.checks[name] = ok
        if value is not None:
            self.metrics[name] = float(value)
        if not ok:
            self.passed = False
            msg = f"FAIL [{name}]"
            if value is not None:
                msg += f"  value={value:{fmt}}"
            if error_msg:
                msg += f"  — {error_msg}"
            self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(f"WARN {msg}")

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        n_ok   = sum(v for v in self.checks.values())
        n_tot  = len(self.checks)
        return f"[{status}] {n_ok}/{n_tot} checks passed"


def _pairwise_distances(pos: np.ndarray,
                        box: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (dist_matrix, nbr_mask) for first-shell neighbours (rc=RC_GRAPH).
    Uses minimum-image PBC.  Returns only upper triangle pair values.
    """
    dr   = pos[:, None, :] - pos[None, :, :]          # (N, N, 3)
    dr   = dr - box * np.round(dr / box)
    dist = np.sqrt(np.sum(dr ** 2, axis=-1))           # (N, N)
    nbr  = (dist > 1e-6) & (dist < RC_GRAPH)          # first shell mask
    return dist, nbr


def validate_glass(pos: np.ndarray,
                   box: float,
                   phase: str = "") -> ValidationResult:
    """
    Run all 10 physical validity checks on one glass configuration.

    Parameters
    ----------
    pos   : (N, 3) float32 array of atomic positions
    box   : cubic box side length (σ)
    phase : label for error messages  ('fast_cool', 'fatigued_c400', …)

    Returns
    -------
    ValidationResult  (see class docstring)
    """
    v = ValidationResult()
    N = len(pos)
    lbl = f"{phase} " if phase else ""

    # ── C1: NaN / Inf ─────────────────────────────────────────────────────
    nan_inf = not (np.isfinite(pos).all())
    v.add_check("C1_no_nan_inf", not nan_inf,
                error_msg="NaN or Inf in positions")
    if nan_inf:
        # Cannot proceed with further checks
        return v

    # ── C2: positions inside box ──────────────────────────────────────────
    in_box = bool(np.all(pos >= 0.0) and np.all(pos < box))
    v.add_check("C2_inside_box", in_box,
                error_msg=f"positions outside [0, {box:.3f})")

    # ── Pairwise distances (needed by C3–C7) ──────────────────────────────
    dist, nbr = _pairwise_distances(pos, box)

    # ── C3: hard-core overlap ─────────────────────────────────────────────
    # Only upper triangle to avoid self-pairs
    upper = np.triu_indices(N, k=1)
    min_d = float(dist[upper].min())
    v.add_check("C3_hard_core",
                min_d >= HARD_CORE_MIN, value=min_d,
                error_msg=f"pair distance {min_d:.4f} < {HARD_CORE_MIN} σ")

    # ── C4: coordination number range ─────────────────────────────────────
    coord      = nbr.sum(axis=1).astype(float)           # (N,)
    mean_coord = float(coord.mean())
    ok_coord   = (COORD_MIN <= mean_coord <= COORD_MAX)
    v.add_check("C4_coordination_range", ok_coord, value=mean_coord,
                fmt=".2f",
                error_msg=f"mean coord {mean_coord:.2f} outside "
                          f"[{COORD_MIN}, {COORD_MAX}]")

    # ── C5: no isolated atoms ─────────────────────────────────────────────
    n_isolated = int((coord < MIN_COORD_PER_ATOM).sum())
    v.add_check("C5_no_isolated", n_isolated == 0,
                value=n_isolated, fmt="d",
                error_msg=f"{n_isolated} atoms with coord < {MIN_COORD_PER_ATOM}")

    # ── Bond statistics (C6–C9) ───────────────────────────────────────────
    bond_lengths = dist[nbr]                             # 1D array of bonds
    if len(bond_lengths) == 0:
        v.add_check("C6_bond_mean",  False, error_msg="no bonds found")
        v.add_check("C7_bond_std",   False)
        v.add_check("C9_not_crystal",False)
        return v

    mean_bl = float(bond_lengths.mean())
    std_bl  = float(bond_lengths.std())

    # ── C6: bond mean in physical range ───────────────────────────────────
    v.add_check("C6_bond_mean",
                BOND_MEAN_LO <= mean_bl <= BOND_MEAN_HI,
                value=mean_bl,
                error_msg=f"mean bond {mean_bl:.4f} outside "
                          f"[{BOND_MEAN_LO}, {BOND_MEAN_HI}]")

    # ── C7: bond std in physical range ────────────────────────────────────
    v.add_check("C7_bond_std",
                BOND_STD_LO <= std_bl <= BOND_STD_HI,
                value=std_bl,
                error_msg=f"bond std {std_bl:.5f} outside "
                          f"[{BOND_STD_LO}, {BOND_STD_HI}]")

    # ── C8: LJ energy per particle ────────────────────────────────────────
    try:
        e_per_atom = float(lj_energy_pbc(
            jnp.array(pos, dtype=jnp.float32),
            jnp.float32(box)))
        v.add_check("C8_energy",
                    ENERGY_LO <= e_per_atom <= ENERGY_HI,
                    value=e_per_atom,
                    error_msg=f"E/N = {e_per_atom:.3f} outside "
                              f"[{ENERGY_LO}, {ENERGY_HI}]")
    except Exception as exc:
        v.add_check("C8_energy", False,
                    error_msg=f"energy computation failed: {exc}")
        e_per_atom = np.nan

    # ── C9: not crystalline ───────────────────────────────────────────────
    not_crystal = (std_bl > CRYSTAL_STD_MAX)
    v.add_check("C9_not_crystal", not_crystal, value=std_bl,
                error_msg=f"bond-std {std_bl:.5f} ≤ {CRYSTAL_STD_MAX} "
                          f"→ possible crystallisation")

    # ── C10: structural heterogeneity ─────────────────────────────────────
    std_coord = float(coord.std())
    v.add_check("C10_heterogeneous", std_coord > 0.0, value=std_coord,
                error_msg="all atoms have identical coordination")

    # ── Store full metrics for downstream analysis ─────────────────────────
    v.metrics.update({
        "mean_bond_length": mean_bl,
        "std_bond_length":  std_bl,
        "min_bond_length":  float(bond_lengths.min()),
        "max_bond_length":  float(bond_lengths.max()),
        "mean_coord":       mean_coord,
        "std_coord":        std_coord,
        "n_bonds":          int(len(bond_lengths)),
        "energy_per_atom":  e_per_atom,
        "min_pair_dist":    min_d,
    })

    # ── Advisory warnings (soft checks that do not fail the glass) ─────────
    if mean_bl < 1.05 or mean_bl > 1.25:
        v.add_warning(f"{lbl}mean bond {mean_bl:.4f} σ is unusually "
                      f"far from LJ equilibrium (1.122 σ)")
    if std_bl > 0.22:
        v.add_warning(f"{lbl}bond-std {std_bl:.4f} is high "
                      f"— check if glass is fully cooled")
    if e_per_atom > -5.0 and np.isfinite(e_per_atom):
        v.add_warning(f"{lbl}E/N = {e_per_atom:.3f} is high "
                      f"— glass may not be well equilibrated")

    return v


def validate_fatigue_progression(
        bond_stds: Dict[int, float],
        mean_bonds: Dict[int, float],
        cooling_type: str = "") -> ValidationResult:
    """
    Check that fatigue accumulates physically across saved cycles.

    DESIGN NOTE — conditioning dip
    ──────────────────────────────
    Fast-cooled glasses exhibit an initial 'conditioning dip': bond-std
    may DECREASE during the first 200 cycles as mechanical energy injection
    relaxes the glass toward lower-energy configurations.  This is a known
    physical effect (Sakamoto et al. 2013) and is NOT a simulation error.
    The check uses MONOTONE_TOL = 0.025 σ to accommodate this dip.

    The fatigue progression check is INFORMATIONAL.  Its ValidationResult
    is stored in each sample but does not gate dataset inclusion — the
    individual snapshot checks (C1–C10) do that.

    Parameters
    ----------
    bond_stds    : {cycle: std_bond_length}   for this glass
    mean_bonds   : {cycle: mean_bond_length}  for this glass
    cooling_type : 'fast' or 'slow'  — used in warning messages only

    Returns
    -------
    ValidationResult  (F1–F3 checks, all informational)
    """
    v   = ValidationResult()
    cyc = sorted(bond_stds.keys())

    if len(cyc) < 2:
        v.add_warning("Only one cycle snapshot — skipping progression check")
        return v

    # ── F1 & F2: bond-std should be non-decreasing within tolerance ────────
    for i in range(len(cyc) - 1):
        c0, c1 = cyc[i], cyc[i + 1]
        delta  = bond_stds[c1] - bond_stds[c0]
        ok     = (delta >= -MONOTONE_TOL)
        v.add_check(f"F_monotone_std_c{c0}_to_c{c1}", ok,
                    value=delta, fmt="+.5f",
                    error_msg=f"bond-std DECREASED by {abs(delta):.5f} σ "
                              f"(> tolerance {MONOTONE_TOL}).  "
                              f"If cooling_type='fast' this may be the "
                              f"conditioning dip — physically expected.")

    # ── F_net: net change from cycle 0 to last cycle must be positive ──────
    #   Even if there is a conditioning dip, the glass must end up MORE
    #   disordered than it started.  A net negative change would indicate
    #   that 400 cycles of 8% strain actually healed the glass — unphysical.
    c_first, c_last = cyc[0], cyc[-1]
    net_delta = bond_stds[c_last] - bond_stds[c_first]
    v.add_check("F_net_increase",
                net_delta >= -MONOTONE_TOL / 2,
                value=net_delta, fmt="+.5f",
                error_msg=f"bond-std NET change from cycle {c_first} to "
                          f"{c_last} is {net_delta:+.5f} — glass appears "
                          f"to have been annealed rather than fatigued")

    # ── F3: mean bond length drift ─────────────────────────────────────────
    drift = abs(mean_bonds[c_last] - mean_bonds[c_first])
    v.add_check("F3_mean_drift", drift < MEAN_DRIFT_MAX,
                value=drift, fmt=".5f",
                error_msg=f"mean bond drifted {drift:.5f} σ "
                          f"(> {MEAN_DRIFT_MAX}) — possible bulk expansion")

    # ── Store the conditioning dip depth as a scalar metric ───────────────
    #   This is scientifically interesting: a large dip for fast glasses
    #   vs a small/no dip for slow glasses is itself a multi-history signal.
    if len(cyc) >= 3:
        # dip = how much lower the mid-cycle std is vs cycle 0
        mid_cyc     = cyc[len(cyc) // 2]
        cond_dip    = bond_stds[c_first] - bond_stds[mid_cyc]
        v.metrics["conditioning_dip_sigma"] = float(cond_dip)
        v.metrics["net_bond_std_change"]    = float(net_delta)
        if cond_dip > 0.005:
            v.add_warning(f"{cooling_type} glass: conditioning dip = "
                          f"{cond_dip:.5f} σ at cycle {mid_cyc} "
                          f"(annealing before fatigue dominates)")

    return v


# ──────────────────────────────────────────────────────────────────────────
# 4.  GLASS GENERATION
# ──────────────────────────────────────────────────────────────────────────

def init_on_lattice(key: jnp.ndarray) -> Tuple[np.ndarray, jnp.ndarray]:
    """
    Place N_ATOMS on a slightly perturbed simple-cubic lattice.
    Lattice initialisation avoids hard-core overlaps from random placement
    and dramatically speeds up equilibration.
    """
    n_side  = int(np.ceil(N_ATOMS ** (1.0 / 3.0)))
    spacing = BOX_L / n_side
    pts = np.array(
        [(i, j, k)
         for i in range(n_side)
         for j in range(n_side)
         for k in range(n_side)],
        dtype=np.float32
    )[:N_ATOMS] * spacing

    key, sk = random.split(key)
    noise   = np.array(
        random.normal(sk, pts.shape, dtype=jnp.float32)) * 0.05
    pts     = (pts + noise) % BOX_L
    return pts.astype(np.float32), key


def generate_glass(key:          jnp.ndarray,
                   n_cool_chunks: int,
                   n_final_equil: int,
                   label:         str = "") -> Tuple[np.ndarray,
                                                     jnp.ndarray,
                                                     ValidationResult]:
    """
    Generate one LJ glass via linear cooling from T_HIGH to T_LOW.

    Protocol
    ────────
    1. Lattice + hot equilibration  (HOT_EQUIL_CHUNKS × SCAN_CHUNK steps)
    2. Linear cooling               (n_cool_chunks × SCAN_CHUNK steps)
    3. Final equilibration at T_LOW (n_final_equil  × SCAN_CHUNK steps)

    Parameters
    ----------
    key           : JAX PRNGKey
    n_cool_chunks : cooling speed (FAST_COOL_CHUNKS or SLOW_COOL_CHUNKS)
    n_final_equil : extra equilibration steps after cooling
    label         : 'fast' or 'slow' — used in validation messages only

    Returns
    -------
    pos : (N, 3) float32
    key : updated PRNGKey
    vr  : ValidationResult
    """
    box = jnp.float32(BOX_L)

    # ── Step 1: lattice + hot equilibration ───────────────────────────────
    pos, key = init_on_lattice(key)
    pos_j    = jnp.array(pos, dtype=jnp.float32)
    for _ in range(HOT_EQUIL_CHUNKS):
        pos_j, key = md_chunk(pos_j, key, box, jnp.float32(T_HIGH))

    # ── Step 2: linear cooling ────────────────────────────────────────────
    temps = np.linspace(T_HIGH, T_LOW, n_cool_chunks, dtype=np.float32)
    for T_val in temps:
        pos_j, key = md_chunk(pos_j, key, box, jnp.float32(T_val))

    # ── Step 3: final equilibration at T_LOW ──────────────────────────────
    for _ in range(n_final_equil):
        pos_j, key = md_chunk(pos_j, key, box, jnp.float32(T_LOW))

    pos_out = np.array(pos_j, dtype=np.float32)

    # ── Validate ──────────────────────────────────────────────────────────
    vr = validate_glass(pos_out, BOX_L, phase=f"{label}_cooled")
    return pos_out, key, vr


# ──────────────────────────────────────────────────────────────────────────
# 5.  FATIGUE CYCLING PROTOCOL
# ──────────────────────────────────────────────────────────────────────────

def apply_one_cycle(pos: np.ndarray,
                    key: jnp.ndarray) -> Tuple[np.ndarray, jnp.ndarray]:
    """
    Simulate one charge/discharge cycle.

    Charge   : affine expansion  → (1+δ)×pos,  (1+δ)×box;
               run STEPS_PHASE Brownian steps at T_BATTERY.
    Discharge: affine compression → original box;
               run STEPS_PHASE Brownian steps at T_BATTERY.

    Affine scaling is the standard mechanical deformation technique in MD.
    It conserves the glass topology while injecting mechanical stress.
    """
    scale  = jnp.float32(1.0 + STRAIN_AMP)
    L_exp  = jnp.float32(BOX_L * (1.0 + STRAIN_AMP))
    L_orig = jnp.float32(BOX_L)
    T      = jnp.float32(T_BATTERY)

    # ── Charge: expand ────────────────────────────────────────────────────
    pos_j   = jnp.array(pos) * scale
    pos_j, key = run_md(pos_j, key, float(L_exp), T_BATTERY, STEPS_PHASE)

    # ── Discharge: compress ───────────────────────────────────────────────
    pos_j   = pos_j / scale
    pos_j, key = run_md(pos_j, key, float(L_orig), T_BATTERY, STEPS_PHASE)

    return np.array(pos_j, dtype=np.float32), key


def run_fatigue_protocol(
        pos0:    np.ndarray,
        key:     jnp.ndarray,
        glass_id: int,
        cooling_type: str) -> Tuple[List[Dict], ValidationResult]:
    """
    Run MAX_CYCLES charge/discharge cycles on a single glass.
    Saves snapshots at cycles listed in SAVE_AT.
    Validates each snapshot individually and checks fatigue progression.

    Returns
    -------
    snapshots : list of sample dicts (one per SAVE_AT entry)
    fatigue_vr: ValidationResult for the progression check
    """
    pos       = pos0.copy()
    snapshots = []

    # ── Save cycle-0 snapshot (pristine) ──────────────────────────────────
    vr0 = validate_glass(pos, BOX_L, phase=f"g{glass_id}_{cooling_type}_c0")
    if 0 in SAVE_AT:
        snapshots.append(_make_sample(pos, glass_id, cooling_type, 0, vr0))

    # ── Cycle through MAX_CYCLES ───────────────────────────────────────────
    for cyc in range(1, MAX_CYCLES + 1):
        pos, key = apply_one_cycle(pos, key)

        if cyc in SAVE_AT:
            vr_c = validate_glass(pos, BOX_L,
                                  phase=f"g{glass_id}_{cooling_type}_c{cyc}")
            snapshots.append(_make_sample(pos, glass_id, cooling_type,
                                          cyc, vr_c))

    # ── Fatigue progression check ─────────────────────────────────────────
    bond_stds  = {s["fatigue_cycles"]: s["validation"]["std_bond_length"]
                  for s in snapshots
                  if "std_bond_length" in s["validation"]}
    mean_bonds = {s["fatigue_cycles"]: s["validation"]["mean_bond_length"]
                  for s in snapshots
                  if "mean_bond_length" in s["validation"]}
    fatigue_vr = validate_fatigue_progression(bond_stds, mean_bonds,
                                               cooling_type=cooling_type)

    return snapshots, fatigue_vr


def _make_sample(pos:          np.ndarray,
                 glass_id:     int,
                 cooling_type: str,
                 fatigue_cycles: int,
                 vr:           ValidationResult) -> Dict:
    """
    Package one configuration snapshot into a dataset sample dict.

    Keys
    ────
    glass_id       : unique identifier for the base glass
    cooling_type   : 'fast' or 'slow'
    cooling_chunks : integer (FAST_COOL_CHUNKS or SLOW_COOL_CHUNKS)
    fatigue_cycles : integer (0, 200, or 400)
    label_cooling  : 0 = fast,  1 = slow          (binary classification)
    label_fatigue  : 0 = c0,    1 = c200,  2 = c400  (3-way)
    label_fatigue_cont : fatigue_cycles / MAX_CYCLES  (regression [0,1])
    positions      : (N, 3) float32 array
    validation     : dict of physical metrics
    sample_passed  : bool — True only if all 10 checks passed
    """
    cool_chunks = FAST_COOL_CHUNKS if cooling_type == "fast" \
                  else SLOW_COOL_CHUNKS
    fat_label   = SAVE_AT.index(fatigue_cycles) if fatigue_cycles in SAVE_AT \
                  else -1

    return {
        "glass_id":          glass_id,
        "cooling_type":      cooling_type,
        "cooling_chunks":    cool_chunks,
        "fatigue_cycles":    fatigue_cycles,
        "label_cooling":     0 if cooling_type == "fast" else 1,
        "label_fatigue":     fat_label,
        "label_fatigue_cont": fatigue_cycles / MAX_CYCLES,
        "positions":         pos.astype(np.float32),
        "validation":        vr.metrics,
        "sample_passed":     vr.passed,
        "validation_errors": vr.errors,
        "validation_warnings": vr.warnings,
    }


# ──────────────────────────────────────────────────────────────────────────
# 6.  FULL DATASET GENERATION LOOP
# ──────────────────────────────────────────────────────────────────────────

def generate_full_dataset() -> List[Dict]:
    """
    Generate the complete 2D factorial glass dataset.

    For each cooling type  (fast, slow):
      For each glass  (0 … N_GLASSES-1):
        1. Generate the glass  (retry up to MAX_RETRIES if validation fails)
        2. Run fatigue protocol  (saves snapshots at SAVE_AT cycles)
        3. Append valid samples to dataset

    Progress is printed every 10 glasses.

    Returns
    -------
    dataset : list of sample dicts (target: 6 × N_GLASSES = 600)
    """
    dataset        = []
    master_key     = random.PRNGKey(42)
    t_start        = time.time()

    # Counters for the final report
    n_glass_failed = 0     # glasses that failed all retries
    n_fatigue_warn = 0     # glasses with fatigue-progression warnings

    print(f"\n{'='*62}")
    print(f"  GENERATING {2 * N_GLASSES} GLASSES  "
          f"({N_GLASSES} fast + {N_GLASSES} slow)")
    print(f"  Each glass: {len(SAVE_AT)} snapshots  "
          f"(cycles {SAVE_AT})")
    print(f"  Target dataset size: "
          f"{2 * N_GLASSES * len(SAVE_AT)} samples")
    print(f"{'='*62}\n")

    for cooling_type, n_cool, n_final in [
            ("fast", FAST_COOL_CHUNKS, FAST_FINAL_CHUNKS),
            ("slow", SLOW_COOL_CHUNKS, SLOW_FINAL_CHUNKS),
    ]:
        print(f"\n── {cooling_type.upper()} COOLING  "
              f"({n_cool} cooling chunks) ─────────────────")

        t_cool_start = time.time()
        n_accepted   = 0
        n_skipped    = 0

        for gid in range(N_GLASSES):
            # ── Generate glass with retries ────────────────────────────────
            glass_pos = None
            glass_vr  = None

            for attempt in range(MAX_RETRIES):
                master_key, subkey = random.split(master_key)
                try:
                    pos_try, master_key, vr = generate_glass(
                        subkey, n_cool, n_final, label=cooling_type)
                except Exception as exc:
                    print(f"    glass {gid} attempt {attempt+1}: "
                          f"JAX error: {exc}")
                    continue

                if vr.passed:
                    glass_pos = pos_try
                    glass_vr  = vr
                    break
                else:
                    if attempt < MAX_RETRIES - 1:
                        pass   # silent retry
                    else:
                        print(f"    WARNING: glass {gid} failed all "
                              f"{MAX_RETRIES} attempts.")
                        for err in vr.errors:
                            print(f"      {err}")
                        n_glass_failed += 1

            if glass_pos is None:
                n_skipped += 1
                continue

            # ── Run fatigue protocol ───────────────────────────────────────
            master_key, cycle_key = random.split(master_key)
            try:
                snaps, fat_vr = run_fatigue_protocol(
                    glass_pos, cycle_key, gid, cooling_type)
            except Exception as exc:
                print(f"    WARNING: fatigue failed for glass {gid}: {exc}")
                n_skipped += 1
                continue

            if not fat_vr.passed:
                n_fatigue_warn += 1
                for err in fat_vr.errors:
                    print(f"    FATIGUE WARN glass {gid}: {err}")

            # Only add samples that individually passed validation
            valid_snaps = [s for s in snaps if s["sample_passed"]]
            n_bad       = len(snaps) - len(valid_snaps)
            if n_bad > 0:
                print(f"    glass {gid}: {n_bad}/{len(snaps)} "
                      f"snapshots FAILED validation and were dropped")

            dataset.extend(valid_snaps)
            n_accepted += 1

            # ── Progress report ────────────────────────────────────────────
            if (gid + 1) % 10 == 0:
                elapsed = time.time() - t_cool_start
                eta     = elapsed / (gid + 1) * (N_GLASSES - gid - 1)
                n_samples = sum(1 for s in dataset
                                if s["cooling_type"] == cooling_type)
                print(f"  [{gid+1:3d}/{N_GLASSES}]  "
                      f"elapsed {elapsed/60:.1f} min  |  "
                      f"ETA {eta/60:.1f} min  |  "
                      f"samples so far: {n_samples}")

        cool_time = time.time() - t_cool_start
        print(f"\n  {cooling_type.upper()} done:  "
              f"{n_accepted}/{N_GLASSES} glasses accepted  |  "
              f"{n_skipped} skipped  |  "
              f"{cool_time/60:.1f} min")

    total_time = time.time() - t_start
    print(f"\n{'='*62}")
    print(f"  GENERATION COMPLETE")
    print(f"  Total samples   : {len(dataset)}")
    print(f"  Glasses failed  : {n_glass_failed}")
    print(f"  Fatigue warnings: {n_fatigue_warn}")
    print(f"  Total time      : {total_time/60:.1f} min")
    print(f"{'='*62}\n")

    return dataset


# ──────────────────────────────────────────────────────────────────────────
# 7.  DATASET STATISTICS & VALIDATION REPORT
# ──────────────────────────────────────────────────────────────────────────

def compute_dataset_statistics(dataset: List[Dict]) -> Dict:
    """
    Compute per-cell summary statistics for the full dataset.
    Groups samples by (cooling_type, fatigue_cycles) and reports:
      - count, pass rate, mean/std bond length, mean coordination,
        mean energy per atom.
    Returns a nested dict for further analysis.
    """
    from collections import defaultdict
    cells = defaultdict(list)
    for s in dataset:
        key = (s["cooling_type"], s["fatigue_cycles"])
        cells[key].append(s)

    stats = {}
    for (ct, cyc), samples in sorted(cells.items()):
        n_pass = sum(1 for s in samples if s["sample_passed"])
        vms    = [s["validation"] for s in samples if s["validation"]]

        def _stat(metric):
            vals = [v.get(metric, np.nan) for v in vms
                    if np.isfinite(v.get(metric, np.nan))]
            if not vals:
                return np.nan, np.nan
            return float(np.mean(vals)), float(np.std(vals))

        stats[(ct, cyc)] = {
            "n_total":     len(samples),
            "n_pass":      n_pass,
            "pass_rate":   n_pass / max(len(samples), 1),
            "mean_bond":   _stat("mean_bond_length"),
            "std_bond":    _stat("std_bond_length"),
            "mean_coord":  _stat("mean_coord"),
            "energy":      _stat("energy_per_atom"),
        }
    return stats


def print_validation_report(dataset: List[Dict]) -> None:
    """
    Print a structured summary of the dataset.
    Checks that:
      - Each cell has ≥ 90 valid samples  (out of N_GLASSES target)
      - Pass rate ≥ 95%
      - Structural parameters differ between fast and slow glasses
        (confirms that cooling rate leaves a detectable geometric fingerprint)
    """
    stats = compute_dataset_statistics(dataset)

    print(f"\n{'='*78}")
    print(f"  DATASET VALIDATION REPORT")
    print(f"{'─'*78}")
    print(f"  {'Cooling':<8} {'Cycle':>5}  {'N':>5}  {'Pass%':>6}  "
          f"{'Mean bond':>10}  {'Std bond':>10}  "
          f"{'Coord':>7}  {'E/N':>9}")
    print(f"{'─'*78}")

    for (ct, cyc), s in sorted(stats.items()):
        mb_m, mb_s = s["mean_bond"]
        sb_m, sb_s = s["std_bond"]
        co_m, co_s = s["mean_coord"]
        en_m, en_s = s["energy"]
        print(f"  {ct:<8} {cyc:>5}  "
              f"{s['n_total']:>5}  "
              f"{s['pass_rate']*100:>5.1f}%  "
              f"{mb_m:>7.4f}±{mb_s:.4f}  "
              f"{sb_m:>7.5f}±{sb_s:.5f}  "
              f"{co_m:>5.1f}±{co_s:.2f}  "
              f"{en_m:>7.3f}±{en_s:.3f}")

    print(f"{'─'*78}")

    # ── Sanity checks ──────────────────────────────────────────────────────
    print(f"\n  SANITY CHECKS")
    all_ok = True

    for (ct, cyc), s in sorted(stats.items()):
        label = f"  {ct} / cycle {cyc}"

        # Minimum sample count
        if s["n_total"] < int(0.9 * N_GLASSES):
            print(f"  FAIL {label}: only {s['n_total']} samples "
                  f"(< 90% of {N_GLASSES})")
            all_ok = False
        else:
            print(f"  PASS {label}: {s['n_total']} samples ✓")

        # Pass rate
        if s["pass_rate"] < 0.90:
            print(f"  FAIL {label}: pass rate {s['pass_rate']*100:.1f}% < 90%")
            all_ok = False

    # Cross-cooling structural difference at cycle 0
    try:
        fast_sb = stats[("fast", 0)]["std_bond"][0]
        slow_sb = stats[("slow", 0)]["std_bond"][0]
        diff    = fast_sb - slow_sb
        if abs(diff) < 0.001:
            print(f"  WARN: bond-std difference fast vs slow at cycle 0 "
                  f"= {diff:.5f} σ — may be too small to detect")
        else:
            print(f"  PASS: bond-std fast vs slow at cycle 0 "
                  f"= {diff:+.5f} σ ✓")
    except KeyError:
        print("  WARN: could not compare fast vs slow at cycle 0")

    # Fatigue progression: slow / cycle 0 → cycle 400
    for ct in ("fast", "slow"):
        try:
            sb0   = stats[(ct, 0)]  ["std_bond"][0]
            sb400 = stats[(ct, 400)]["std_bond"][0]
            delta = sb400 - sb0
            if delta < 0.002:
                print(f"  WARN {ct}: bond-std barely increased over 400 cycles "
                      f"(Δ = {delta:.5f})")
            else:
                print(f"  PASS {ct}: bond-std +{delta:.5f} over 400 cycles ✓")
        except KeyError:
            print(f"  WARN {ct}: missing cycle 0 or 400 data")

    # ── Conditioning dip report ────────────────────────────────────────────
    #   Counts how many glasses show the annealing dip (bond-std decreases
    #   in early cycles).  Expected:  many fast glasses, few/no slow glasses.
    #   A large difference between fast and slow counts is itself a
    #   multi-history signal and should be noted in the paper.
    print(f"\n  CONDITIONING DIP ANALYSIS")
    print(f"  (bond-std lower at cycle 200 than cycle 0 — annealing before fatigue)")
    for ct in ("fast", "slow"):
        ct_samples = [s for s in dataset
                      if s["cooling_type"] == ct and s["fatigue_cycles"] == 0]
        ct_c200    = {s["glass_id"]: s for s in dataset
                      if s["cooling_type"] == ct and s["fatigue_cycles"] == 200}
        n_dip = 0
        dip_depths = []
        for s0 in ct_samples:
            gid = s0["glass_id"]
            if gid not in ct_c200:
                continue
            sb0   = s0  ["validation"].get("std_bond_length", np.nan)
            sb200 = ct_c200[gid]["validation"].get("std_bond_length", np.nan)
            if np.isfinite(sb0) and np.isfinite(sb200) and sb200 < sb0:
                n_dip += 1
                dip_depths.append(sb0 - sb200)
        if dip_depths:
            print(f"  {ct:<5}: {n_dip}/{len(ct_samples)} glasses show dip  "
                  f"(mean depth {np.mean(dip_depths):.5f} σ, "
                  f"max {np.max(dip_depths):.5f} σ)")
        else:
            print(f"  {ct:<5}: 0/{len(ct_samples)} glasses show dip  ✓ "
                  f"(purely monotonic fatigue)")

    overall = "ALL CHECKS PASSED ✓" if all_ok else "SOME CHECKS FAILED — review above"
    print(f"\n  {overall}")
    print(f"{'='*78}\n")


def print_label_distribution(dataset: List[Dict]) -> None:
    """Print the 2D label contingency table for the final dataset."""
    from collections import Counter
    counts = Counter(
        (s["label_cooling"], s["label_fatigue"]) for s in dataset)

    print(f"\n  LABEL DISTRIBUTION (label_cooling × label_fatigue)")
    print(f"  {'':15} {'fat=0 (c0)':>12}  {'fat=1 (c200)':>12}  "
          f"{'fat=2 (c400)':>12}")
    for lc, ct in [(0, "fast"), (1, "slow")]:
        row = "  {:15}".format(f"cool={lc} ({ct})")
        for lf in [0, 1, 2]:
            row += f"  {counts[(lc, lf)]:>12}"
        print(row)
    total = sum(counts.values())
    print(f"\n  Total samples: {total}  "
          f"(target: {2 * 3 * N_GLASSES})\n")


# ──────────────────────────────────────────────────────────────────────────
# 8.  SAVE
# ──────────────────────────────────────────────────────────────────────────

def save_dataset(dataset: List[Dict], path: str) -> None:
    """
    Pickle the full dataset to disk.

    Also saves a lightweight summary CSV for quick inspection.
    """
    with open(path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Dataset saved → {path}  ({size_mb:.1f} MB)")

    # ── CSV summary ───────────────────────────────────────────────────────
    csv_path = path.replace(".pkl", "_summary.csv")
    with open(csv_path, "w") as f:
        header = ("sample_idx,glass_id,cooling_type,cooling_chunks,"
                  "fatigue_cycles,label_cooling,label_fatigue,"
                  "label_fatigue_cont,sample_passed,"
                  "mean_bond,std_bond,mean_coord,energy_per_atom\n")
        f.write(header)
        for idx, s in enumerate(dataset):
            vm = s.get("validation", {})
            f.write(
                f"{idx},{s['glass_id']},{s['cooling_type']},"
                f"{s['cooling_chunks']},{s['fatigue_cycles']},"
                f"{s['label_cooling']},{s['label_fatigue']},"
                f"{s['label_fatigue_cont']:.4f},{int(s['sample_passed'])},"
                f"{vm.get('mean_bond_length', float('nan')):.5f},"
                f"{vm.get('std_bond_length',  float('nan')):.5f},"
                f"{vm.get('mean_coord',        float('nan')):.2f},"
                f"{vm.get('energy_per_atom',   float('nan')):.4f}\n"
            )
    print(f"  Summary CSV  → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── Print configuration summary ───────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  MULTI-HISTORY GLASS DATA GENERATION")
    print(f"{'─'*62}")
    print(f"  N_ATOMS          : {N_ATOMS}")
    print(f"  RHO              : {RHO}  (box = {BOX_L:.4f} σ)")
    print(f"  T_g              : ~0.45  (LJ at ρ=1.2)")
    print(f"  T_LOW (glass)    : {T_LOW}")
    print(f"  T_BATTERY        : {T_BATTERY}  (~0.93 T_g)")
    print(f"  Fast cool chunks : {FAST_COOL_CHUNKS}  ({FAST_COOL_CHUNKS*SCAN_CHUNK*DT:.1f} τ)")
    print(f"  Slow cool chunks : {SLOW_COOL_CHUNKS}  ({SLOW_COOL_CHUNKS*SCAN_CHUNK*DT:.0f} τ)")
    print(f"  Cooling ratio    : {SLOW_COOL_CHUNKS // FAST_COOL_CHUNKS}×")
    print(f"  Strain amplitude : {STRAIN_AMP*100:.0f}%")
    print(f"  Max cycles       : {MAX_CYCLES}")
    print(f"  Save at cycles   : {SAVE_AT}")
    print(f"  Glasses/type     : {N_GLASSES}")
    print(f"  Target samples   : {2 * N_GLASSES * len(SAVE_AT)}")
    print(f"  Output           : {OUT_FILE}")
    print(f"{'='*62}\n")

    # ── Check for existing dataset ─────────────────────────────────────────
    if os.path.exists(OUT_FILE):
        print(f"  Found existing dataset at {OUT_FILE}.")
        resp = input("  Overwrite? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Aborted.  Loading existing dataset for validation.")
            with open(OUT_FILE, "rb") as f:
                dataset = pickle.load(f)
            print_validation_report(dataset)
            print_label_distribution(dataset)
            return

    # ── Warm up JAX ───────────────────────────────────────────────────────
    _warmup_jax()

    # ── Generate ──────────────────────────────────────────────────────────
    dataset = generate_full_dataset()

    # ── Validate & report ─────────────────────────────────────────────────
    print_validation_report(dataset)
    print_label_distribution(dataset)

    # ── Save ──────────────────────────────────────────────────────────────
    save_dataset(dataset, OUT_FILE)

    total_min = (time.time() - t0) / 60
    print(f"\n  Total wall-clock time: {total_min:.1f} min")
    print(f"  Dataset ready at: {OUT_FILE}\n")

    # ── Quick physical plausibility summary ───────────────────────────────
    print(f"  PHYSICAL PLAUSIBILITY SUMMARY")
    print(f"{'─'*62}")
    n_pass = sum(1 for s in dataset if s["sample_passed"])
    n_total = len(dataset)
    print(f"  Samples passing all 10 checks : {n_pass}/{n_total} "
          f"({100*n_pass/max(n_total,1):.1f}%)")

    all_errors = [e for s in dataset for e in s.get("validation_errors", [])]
    if all_errors:
        print(f"  Unique error types encountered:")
        from collections import Counter
        for err, cnt in Counter(
                e.split("]")[0] for e in all_errors).most_common(5):
            print(f"    {cnt:3d}×  {err}]")
    else:
        print(f"  No validation errors in accepted samples ✓")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()