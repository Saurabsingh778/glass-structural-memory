# Glass Structural Memory: Geometric Orthogonality of Competing Processing Histories

**Paper:** "Geometric Orthogonality of Competing Structural Memories in Glass:
Temperature Controls Mechanical Erasure of Thermal History"
**Author:** Saurab Singh (Independent Researcher) — saurabsingh778@gmail.com
**Preprint:** [ChemRxiv / Zenodo link here]

---

## What This Repository Contains

Code and data for a systematic computational study of how competing
processing histories — thermal (cooling rate) and mechanical (cyclic
fatigue) — are encoded in the local bond-length geometry of a
Lennard-Jones model glass, and how one history erases the other.

---

## Key Results

| Result | Finding |
|--------|---------|
| Geometric orthogonality | Both histories occupy distinct PCA directions in GNN latent space (cross-contamination < 0.21) in all 9 conditions |
| Simultaneous decoding | GATv2 recovers both histories from one snapshot with < 3pp accuracy loss vs single-task baselines |
| Temperature controls interference | Cooling-rate accuracy drop after 400 fatigue cycles: 14–31 pp across T=0.78–1.02 Tg; strain amplitude has no effect |
| Palimpsest / Amnesia crossover | Below Tg: fatigue drives fast-cooled glasses toward slow-cooled attractor (α=+0.24 to +0.46). Above Tg: symmetric erasure (α≈0). Sharp crossover at Tg |
| Super-Arrhenius memory decay | Thermal memory timescale τ follows VTF law: T₀=0.33, R²=0.999. KWW β increases 0.41→0.88 toward Tg |

---

## Reproduction

### 2. Generate glass data (one temperature condition)
```bash
python data_en.py
# Edit T_BATTERY and STRAIN_AMP at top of file for each condition
```

### 3. Train GATv2 (all experiments for one condition)
```bash
python train.py
```

### 4. Asymmetry analysis (all 9 conditions)
```bash
python asymmetry_analysis.py \
    --data_dir ./data \
    --out_dir  ./asymmetry_results
```

### 5. Forgetting curve + VTF fit
```bash
python dense_arrhenius.py \
    --dense_dir ./data \
    --cache_pkl ./arrhenius_results/forgetting_curves.pkl \
    --out_dir   ./dense_arrhenius_results
```

---

## Hardware

All simulations run on free-tier cloud GPU (Kaggle Tesla T4) and a
local NVIDIA RTX 4060 Laptop GPU using JAX-MD and PyTorch Geometric.
Total compute: approximately 12 GPU-hours for all 9 conditions.

---

## Dependencies

- JAX ≥ 0.7
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.4
- NumPy, SciPy, scikit-learn, matplotlib, pandas

---

## Citation

If you use this code or data, please cite:
```bibtex
@article{singh2026glass,
  title   = {Geometric Orthogonality of Competing Structural Memories
             in Glass: Temperature Controls Mechanical Erasure of
             Thermal History},
  author  = {Singh, Saurab},
  year    = {2026},
  note    = {Preprint}
}
```

---

## Related Work

- Paper 1 (thermal history): [Zenodo DOI: 10.5281/zenodo.18870490]
- Paper 2 (cyclic fatigue): [Zenodo DOI: 10.5281/zenodo.18910005]
```

---

## GitHub Topics to Add

Set these in the repo settings under Topics:
```
glass-physics  graph-neural-networks  molecular-dynamics  
materials-science  amorphous-solids  gnn  jax  pytorch-geometric  
structural-memory  battery-materials