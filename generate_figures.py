#!/usr/bin/env python3
"""
Updated figure generation — adds T=0.40 and T=0.44 to the temperature sweep.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

os.makedirs("paper_figures_new", exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.6,
    "lines.markersize":   7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
})

C_COOL   = "#c0392b"
C_STRAIN = "#2980b9"
C_ORTH   = "#27ae60"
C_MULTI  = "#8e44ad"
C_SINGLE = "#e67e22"
C_FAST   = "#e74c3c"
C_SLOW   = "#3498db"
GREY     = "#7f8c8d"
LIGHT    = "#ecf0f1"

TG = 0.45   # glass transition temperature (LJ units)

# ── Strain sweep (T=0.42) ─────────────────────────────────────────────────
run_A = {
    "x":        [4,    6,    8,    12],
    "drop":     [24,   20,   25,   24],
    "drop_std": [6.32, 5.48, 10.49, 10.44],
    "acc4A":    [94.0, 92.5, 95.0, 95.5],
    "acc4A_std":[2.55, 4.18, 3.16, 1.00],
    "acc4B":    [70.0, 72.5, 70.0, 71.5],
    "acc4B_std":[6.32, 5.48, 10.49, 10.44],
    "acc1A":    [81.5, 79.8, 80.3, 80.0],
    "acc1A_std":[3.39, 3.74, 4.96, 2.24],
    "acc2C":    [81.8, 80.7, 79.7, 77.7],
    "acc2C_std":[2.91, 1.93, 3.32, 1.78],
    "xcontam":  [0.0448, 0.0761, 0.2003, 0.1308],
    "r2_fast":  [0.6932, 0.6945, 0.7050, 0.7190],
    "r2_slow":  [0.7128, 0.6936, 0.7292, 0.6887],
    "r2_all":   [0.7259, 0.7141, 0.7200, 0.7361],
    # Feature importance (baseline δ=8%)
    "imp_cool_8": [0.1942, 0.0711, 0.0342, 0.0075, 0.1883, 0.0344, 0.0419, 0.0825],
    "imp_fat_8":  [0.3275, 0.0010, 0.0280, 0.0056, 0.0105, 0.0035, 0.1038, 0.0421],
}

# ── Temperature sweep (δ=8%) — NOW 6 POINTS ──────────────────────────────
# T=0.40: drop=20pp, χ=0.0962, 4A=93.5%, 4B=73.5%
# T=0.44: drop=19pp, χ=0.1520, 4A=93.5%, 4B=74.5%
run_B = {
    "x":        [0.35,  0.38,  0.40,  0.42,  0.44,  0.46],
    "T_Tg":     [0.78,  0.84,  0.89,  0.93,  0.98,  1.02],
    "drop":     [14,    27,    20,    25,    19,    31],
    "drop_std": [9.27,  6.40,  6.44,  10.49, 8.72,  12.29],
    "acc4A":    [92.0,  95.0,  93.5,  95.0,  93.5,  95.5],
    "acc4A_std":[2.92,  2.74,  4.06,  3.16,  3.74,  2.45],
    "acc4B":    [78.0,  68.0,  73.5,  70.0,  74.5,  64.5],
    "acc4B_std":[9.27,  6.40,  6.44,  10.49, 8.72,  12.29],
    "acc1A":    [82.8,  82.2,  81.2,  80.3,  81.8,  77.3],
    "acc1A_std":[2.01,  2.21,  2.01,  4.96,  4.78,  2.44],
    "acc2C":    [81.7,  82.3,  80.7,  79.7,  81.7,  74.5],
    "acc2C_std":[1.67,  3.00,  2.44,  3.32,  1.90,  5.13],
    "xcontam":  [0.0092, 0.1725, 0.0962, 0.2003, 0.1520, 0.0407],
    "r2_fast":  [0.6270, 0.6897, 0.6631, 0.7050, 0.7118, 0.7138],
    "r2_slow":  [0.7084, 0.6936, 0.7005, 0.7292, 0.7255, 0.7249],
    "r2_all":   [0.6734, 0.7080, 0.7204, 0.7200, 0.7293, 0.7164],
    # Feature importance at T=0.35 (deep glass — most distinctive)
    "imp_cool_35": [0.0078, 0.0031, 0.0047, 0.0158, 0.0056, 0.0067, 0.0111, 0.0036],
    "imp_fat_35":  [0.5124,-0.0109, 0.0144, 0.1228, 0.3031, 0.2591, 0.0727, 0.0261],
}

FEAT_SHORT = ["r̄", "σ_r", "r_min", "r_max", "coord", "skew", "Q25", "Q75"]


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Interference map
# ══════════════════════════════════════════════════════════════════════════
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    # Panel A — temperature
    ax = axes[0]
    x   = np.array(run_B["x"])
    y   = np.array(run_B["drop"], dtype=float)
    ye  = np.array(run_B["drop_std"])

    ax.errorbar(x, y, yerr=ye, fmt="o-", color=C_COOL,
                capsize=4, capthick=1.2, elinewidth=1.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    ax.axvspan(0.33, 0.405, color=C_COOL,    alpha=0.06, label="Deep glass\n(elastic)")
    ax.axvspan(0.405, 0.445, color="#f39c12", alpha=0.08, label="Cooperative\nrearrangements")
    ax.axvspan(0.445, 0.49, color=C_STRAIN,  alpha=0.06, label="Above $T_g$\n(liquid)")
    ax.axvline(TG, ls="--", lw=0.9, color=GREY, zorder=1)
    ax.text(TG+0.001, 31, "$T_g$", fontsize=9, color=GREY)

    # Annotate peak
    imax = int(np.argmax(y))
    ax.annotate(f"peak\n{y[imax]:.0f} pp",
                xy=(x[imax], y[imax]),
                xytext=(x[imax]+0.028, y[imax]+1.0),
                fontsize=8, color=C_COOL,
                arrowprops=dict(arrowstyle="->", color=C_COOL, lw=0.9))

    # T/Tg secondary axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(run_B["x"])
    ax2.set_xticklabels([f"{t:.2f}" for t in run_B["T_Tg"]], fontsize=7.5, rotation=30)
    ax2.set_xlabel("$T / T_g$", fontsize=10)

    ax.set_xlabel("Cycling temperature $T$ (LJ units)")
    ax.set_ylabel("Thermal memory loss (pp)")
    ax.set_title("(A) Temperature controls interference", fontweight="bold")
    ax.set_ylim(5, 38)
    ax.set_xlim(0.33, 0.49)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)

    # Panel B — strain
    ax = axes[1]
    x  = np.array(run_A["x"], dtype=float)
    y  = np.array(run_A["drop"], dtype=float)
    ye = np.array(run_A["drop_std"])

    ax.errorbar(x, y, yerr=ye, fmt="s-", color=C_STRAIN,
                capsize=4, capthick=1.2, elinewidth=1.0,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    flat = np.mean(y)
    ax.axhline(flat, ls="--", lw=1.0, color=GREY,
               label=f"Mean = {flat:.0f} pp (no trend)")
    ax.axvspan(3, 8.5, color="#27ae60", alpha=0.07,
               label="LiPON operating range\n(3–8%)")

    ax.set_xlabel("Volumetric strain amplitude δ (%)")
    ax.set_ylabel("Thermal memory loss (pp)")
    ax.set_title("(B) Strain amplitude does not control interference", fontweight="bold")
    ax.set_ylim(5, 38)
    ax.set_xlim(2, 14)
    ax.set_xticks([4, 6, 8, 12])
    ax.legend(loc="upper right", framealpha=0.85, fontsize=8)

    fig.text(0.52, -0.04,
             "Memory loss = acc(pristine) − acc(fatigued c400)  [Exp 4, 5-fold CV]",
             ha="center", fontsize=9, color=GREY, style="italic")

    plt.tight_layout(w_pad=3.0)
    for ext in ["pdf", "png"]:
        plt.savefig(f"paper_figures_new/fig1_interference_map.{ext}")
    plt.close()
    print("  Fig 1 saved.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Multi-task vs single-task
# ══════════════════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    def _panel(ax, run, x_vals, x_label, title):
        x = np.array(x_vals, dtype=float)
        single   = np.array(run["acc1A"])
        single_s = np.array(run["acc1A_std"])
        multi    = np.array(run["acc2C"])
        multi_s  = np.array(run["acc2C_std"])

        ax.errorbar(x - 0.005, single, yerr=single_s, fmt="o--",
                    color=C_SINGLE, capsize=3, capthick=1,
                    label="Single-task", zorder=4)
        ax.errorbar(x + 0.005, multi,  yerr=multi_s,  fmt="s-",
                    color=C_MULTI,  capsize=3, capthick=1,
                    label="Multi-task",  zorder=4)

        diff = single - multi
        for xi, s, m, d in zip(x, single, multi, diff):
            col = "#e74c3c" if d > 2 else "#27ae60"
            ax.annotate(f"{d:+.1f}", xy=(xi, min(s, m) - 1.8),
                        ha="center", fontsize=7.5, color=col)

        ax.axhline(80, ls=":", lw=0.7, color=GREY)
        ax.axhline(50, ls=":", lw=0.7, color=GREY, alpha=0.5)
        ax.text(x[-1] + (0.005 if len(x) == 4 else 0.003), 80.5,
                "80%",    fontsize=8, color=GREY)
        ax.text(x[-1] + (0.005 if len(x) == 4 else 0.003), 50.5,
                "chance", fontsize=8, color=GREY)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cooling-rate accuracy (%)")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(42, 102)
        ax.legend(framealpha=0.85)

    _panel(axes[0], run_B, run_B["x"],
           "Cycling temperature $T$ (LJ units)",
           "(A) Temperature sweep  (δ = 8%)")
    axes[0].set_xticks(run_B["x"])
    axes[0].set_xticklabels([str(t) for t in run_B["x"]], fontsize=8)

    _panel(axes[1], run_A, run_A["x"],
           "Strain amplitude δ (%)",
           "(B) Strain sweep  (T = 0.42)")
    axes[1].set_xticks(run_A["x"])

    fig.text(0.52, -0.04,
             "Numbers above markers = single-task − multi-task accuracy gap",
             ha="center", fontsize=9, color=GREY, style="italic")

    plt.tight_layout(w_pad=3.0)
    for ext in ["pdf", "png"]:
        plt.savefig(f"paper_figures_new/fig2_multitask_accuracy.{ext}")
    plt.close()
    print("  Fig 2 saved.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Latent space orthogonality  (9 conditions total)
# ══════════════════════════════════════════════════════════════════════════
def fig3():
    THRESH_S = 0.15
    THRESH_P = 0.30

    fig = plt.figure(figsize=(13, 4.2))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel A — temperature (6 bars)
    x  = run_B["x"]
    xc = np.array(run_B["xcontam"])
    col_b = [C_ORTH if v < THRESH_S else C_COOL if v < THRESH_P else GREY
             for v in xc]
    ax1.bar(range(len(x)), xc, color=col_b, edgecolor="white",
            linewidth=0.5, alpha=0.88)
    ax1.axhline(THRESH_S, ls="--", lw=1.0, color=C_ORTH,
                label="Strong threshold (0.15)")
    ax1.axhline(THRESH_P, ls=":",  lw=1.0, color=C_COOL,
                label="Partial threshold (0.30)")
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels([f"T={v}" for v in x], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Cross-contamination")
    ax1.set_title("(A) Temperature sweep\n(δ = 8%)", fontweight="bold")
    ax1.set_ylim(0, 0.35)
    ax1.legend(fontsize=8, framealpha=0.85)

    # Panel B — strain (4 bars)
    x2  = run_A["x"]
    xc2 = np.array(run_A["xcontam"])
    col_a = [C_ORTH if v < THRESH_S else C_COOL if v < THRESH_P else GREY
              for v in xc2]
    ax2.bar(range(len(x2)), xc2, color=col_a, edgecolor="white",
            linewidth=0.5, alpha=0.88)
    ax2.axhline(THRESH_S, ls="--", lw=1.0, color=C_ORTH)
    ax2.axhline(THRESH_P, ls=":",  lw=1.0, color=C_COOL)
    ax2.set_xticks(range(len(x2)))
    ax2.set_xticklabels([f"δ={v}%" for v in x2], rotation=25, ha="right")
    ax2.set_ylabel("Cross-contamination")
    ax2.set_title("(B) Strain sweep\n(T = 0.42)", fontweight="bold")
    ax2.set_ylim(0, 0.35)

    # Panel C — all 9 conditions scatter
    all_labels = ["T=0.35", "T=0.38", "T=0.40", "T=0.42\n(base)",
                  "T=0.44", "T=0.46",
                  "δ=4%",   "δ=6%",   "δ=12%"]
    all_xc   = list(run_B["xcontam"]) + [run_A["xcontam"][i] for i in [0,1,3]]
    all_drop = list(run_B["drop"])    + [run_A["drop"][i]    for i in [0,1,3]]
    all_cols = [C_COOL]*6 + [C_STRAIN]*3

    ax3.scatter(all_xc, all_drop, c=all_cols, s=80, zorder=5,
                edgecolors="white", linewidths=0.8)

    offsets = {
        "T=0.35":      (0.005,  1.5),
        "T=0.38":      (0.005, -3.2),
        "T=0.40":      (0.005,  1.5),
        "T=0.42\n(base)": (-0.038, 1.5),
        "T=0.44":      (0.005, -3.2),
        "T=0.46":      (-0.025,  1.5),
        "δ=4%":        (0.005,  1.5),
        "δ=6%":        (0.005, -3.2),
        "δ=12%":       (0.005,  1.5),
    }
    for lbl, xc_v, d in zip(all_labels, all_xc, all_drop):
        ox, oy = offsets.get(lbl, (0.005, 1.5))
        ax3.annotate(lbl, xy=(xc_v, d),
                     xytext=(xc_v + ox, d + oy),
                     fontsize=7.2)

    ax3.axvline(THRESH_S, ls="--", lw=0.9, color=C_ORTH, alpha=0.7)
    ax3.axvline(THRESH_P, ls=":",  lw=0.9, color=C_COOL, alpha=0.7)
    ax3.text(THRESH_S+0.003, 11, "strong",  fontsize=7, color=C_ORTH)
    ax3.text(THRESH_P+0.003, 11, "partial", fontsize=7, color=C_COOL)

    leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=C_COOL,
                  markersize=8, label="Temperature sweep"),
           Line2D([0],[0], marker="o", color="w", markerfacecolor=C_STRAIN,
                  markersize=8, label="Strain sweep")]
    ax3.legend(handles=leg, fontsize=8, framealpha=0.85)
    ax3.set_xlabel("Cross-contamination")
    ax3.set_ylabel("Thermal memory loss (pp)")
    ax3.set_title("(C) All 9 conditions", fontweight="bold")
    ax3.set_xlim(-0.01, 0.28)
    ax3.set_ylim(8, 36)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(f"paper_figures_new/fig3_orthogonality.{ext}")
    plt.close()
    print("  Fig 3 saved.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Feature importance
# ══════════════════════════════════════════════════════════════════════════
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    def _bar(ax, drops, color, title):
        drops = np.array(drops)
        order = np.argsort(drops)[::-1]
        names = [FEAT_SHORT[i] for i in order]
        vals  = drops[order]
        bar_c = [color if v >= 0 else GREY for v in vals]
        bars  = ax.barh(range(len(vals)), vals, color=bar_c,
                        alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, lw=0.8, color="black")
        ax.set_xlabel("AUC drop when feature zeroed")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax.text(v + 0.003 if v >= 0 else v - 0.003,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.5, color="#2c3e50")

    def _rank_diff(cd, fd):
        co = np.argsort(np.argsort(np.array(cd))[::-1])
        fo = np.argsort(np.argsort(np.array(fd))[::-1])
        return np.mean(np.abs(co - fo))

    panels = [
        (axes[0,0], run_A["imp_cool_8"], C_COOL,   "Cooling classifier\n(δ=8%, T=0.42 baseline)"),
        (axes[0,1], run_A["imp_fat_8"],  C_STRAIN,  "Fatigue classifier\n(δ=8%, T=0.42 baseline)"),
        (axes[1,0], run_B["imp_cool_35"], C_COOL,   "Cooling classifier\n(δ=8%, T=0.35 deep glass)"),
        (axes[1,1], run_B["imp_fat_35"],  C_STRAIN,  "Fatigue classifier\n(δ=8%, T=0.35 deep glass)"),
    ]
    for ax, drops, col, title in panels:
        _bar(ax, drops, col, title)

    for row, (cd, fd) in enumerate([
            (run_A["imp_cool_8"],  run_A["imp_fat_8"]),
            (run_B["imp_cool_35"], run_B["imp_fat_35"])]):
        rd = _rank_diff(cd, fd)
        verdict = "DIFFERENT" if rd >= 2.5 else "SIMILAR"
        col = C_ORTH if verdict == "DIFFERENT" else C_COOL
        axes[row,1].text(1.06, 0.5,
            f"Mean rank\ndiff = {rd:.2f}\n→ {verdict}\nfeatures",
            transform=axes[row,1].transAxes,
            va="center", ha="left", fontsize=8.5, color=col,
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=col, lw=0.8))

    fig.suptitle("Feature Importance: Cooling vs Fatigue Classifiers\n"
                 "(AUC drop when feature zeroed across all nodes)",
                 fontsize=11, fontweight="bold", y=1.01)

    for ext in ["pdf", "png"]:
        plt.savefig(f"paper_figures_new/fig4_feature_importance.{ext}")
    plt.close()
    print("  Fig 4 saved.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Regression R² and fast-slow asymmetry  (9 conditions)
# ══════════════════════════════════════════════════════════════════════════
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    def _r2_panel(ax, x, fast, slow, all_, xlabel, title, xticks=None):
        x = np.array(x, dtype=float)
        ax.plot(x, fast, "o--", color=C_FAST,    label="Fast glass",
                markeredgecolor="white", markeredgewidth=0.7)
        ax.plot(x, slow, "s--", color=C_SLOW,    label="Slow glass",
                markeredgecolor="white", markeredgewidth=0.7)
        ax.plot(x, all_, "D-",  color="#2c3e50",  label="All glasses",
                markeredgecolor="white", markeredgewidth=0.7, lw=1.0)
        ax.axhline(0.5, ls=":", lw=0.7, color=GREY, alpha=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("Regression R²")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.45, 0.82)
        ax.legend(fontsize=8, framealpha=0.85)
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks], fontsize=8)

    _r2_panel(axes[0], run_B["x"],
              run_B["r2_fast"], run_B["r2_slow"], run_B["r2_all"],
              "Cycling temperature $T$",
              "(A) Temperature sweep  (δ = 8%)",
              xticks=run_B["x"])

    _r2_panel(axes[1], run_A["x"],
              run_A["r2_fast"], run_A["r2_slow"], run_A["r2_all"],
              "Strain amplitude δ (%)",
              "(B) Strain sweep  (T = 0.42)",
              xticks=run_A["x"])

    # Panel C — all 9 conditions difference
    ax = axes[2]
    labels_all = ["T=0.35","T=0.38","T=0.40","T=0.42\nbase","T=0.44","T=0.46",
                  "δ=4%","δ=6%","δ=12%"]
    fast_all = list(run_B["r2_fast"]) + [run_A["r2_fast"][i] for i in [0,1,3]]
    slow_all = list(run_B["r2_slow"]) + [run_A["r2_slow"][i] for i in [0,1,3]]
    diff_all = np.array(fast_all) - np.array(slow_all)

    colors_c = [C_FAST if d > 0.05 else C_SLOW if d < -0.05 else GREY
                for d in diff_all]
    ax.bar(range(len(labels_all)), diff_all, color=colors_c,
           edgecolor="white", linewidth=0.5, alpha=0.88)
    ax.axhline(0,     lw=0.9, color="black")
    ax.axhline( 0.05, ls="--", lw=0.7, color=C_FAST, alpha=0.6)
    ax.axhline(-0.05, ls="--", lw=0.7, color=C_SLOW, alpha=0.6)
    ax.set_xticks(range(len(labels_all)))
    ax.set_xticklabels(labels_all, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R²(fast) − R²(slow)")
    ax.set_title("(C) Fast−Slow asymmetry\nacross all conditions", fontweight="bold")
    ax.set_ylim(-0.12, 0.12)

    # Annotate T=0.35 (only significant asymmetry)
    i35 = 0
    ax.annotate("T=0.35\nsignificant\nasymmetry",
                xy=(i35, diff_all[i35]),
                xytext=(i35+1.3, diff_all[i35]-0.03),
                fontsize=7.5, color=C_SLOW,
                arrowprops=dict(arrowstyle="->", color=C_SLOW, lw=0.8))

    leg = [Line2D([0],[0], color=C_FAST, lw=2.5, label="Fast > Slow"),
           Line2D([0],[0], color=C_SLOW, lw=2.5, label="Slow > Fast"),
           Line2D([0],[0], color=GREY,   lw=2.5, label="No asymmetry")]
    ax.legend(handles=leg, fontsize=8, framealpha=0.85)

    plt.tight_layout(w_pad=2.5)
    for ext in ["pdf", "png"]:
        plt.savefig(f"paper_figures_new/fig5_regression.{ext}")
    plt.close()
    print("  Fig 5 saved.")


if __name__ == "__main__":
    print("Generating figures...\n")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("\nDone. Output: paper_figures_new/")