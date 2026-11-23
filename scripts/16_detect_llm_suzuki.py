#!/usr/bin/env python3
"""
PCA trajectory grid -- ROWS = SEEDS, COLUMNS = METHODS, one figure per lambda.

Dedicated to a single task (olympus_suzuki). For each lambda in
LAMBDA_LIST, produces ONE figure laid out as:

    rows    = SEEDS       (10 by default: seed 0 .. seed 9)
    columns = METHODS     (gp, lgbo, llm, online)

so a 10-seed sweep gives a 10x4 grid of panels. Each panel shows ONLY its
own method's trajectory (not overlaid with the others) against the
contour background -- same panel style as 3_PCA_trajectory_olympus_suzuki.py,
just with seeds stacked as rows instead of being a separate figure each.

The PCA rotation is fit independently per ROW (per seed) on that seed's
PCA_FIT_METHODS trajectories -- so "PC 1"/"PC 2" mean something different
row to row (same caveat as fitting per-row PCA elsewhere in this repo).
All 4 panels in a given row DO share that row's rotation, since they're
transforms of the same seed's data.

Run from the project root:
    python 5_PCA_trajectory_olympus_suzuki_seed_rows.py
"""
import os
import sys
import re
import string
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["savefig.dpi"] = 600
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA

# ── Project root & imports ─────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
TASK_NAME  = "olympus_suzuki"
TASK_LABEL = "Olympus Suzuki"

# Methods used ONLY to define each row's PCA rotation (not plotted unless
# they also appear in METHODS below).
PCA_FIT_METHODS = ["gp", "llambo", "lgbo"]

# Methods actually PLOTTED as columns, one panel each, per row/seed.
METHODS    = ["gp", "lgbo", "llm", "online"]
ONLINE_ARMS_SUFFIX = None        # None -> original 4-arm pool (no suffix)

# MODEL      = "openai-5.4-mini"
MODEL = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7

SEEDS       = list(range(10))               # 0..9 -- one row each
# LAMBDA_LIST = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0]
LAMBDA_LIST = [10.0]

OPT_COLORS = {"gp": "#5F5E5A", "llm": "#378ADD", "llambo": "#1D9E75", "lgbo": "#D85A30", "online": "#C0392B"}
OPT_LABELS = {"gp": "GP", "llm": "LLM", "llambo": "LLAMBO", "lgbo": "LGBO", "online": "HybridBO"}

ANNOTATE   = False
GRID_N     = 150   # lower than the single-figure scripts -- up to 10x more contour computes
PCA_PAD    = 0.08

# ── figure quality / typography ────────────────────────────────────────────
DPI            = 600
PANEL_W        = 4.5
PANEL_H        = 3.8
LABEL_FONTSIZE = 16   # y-axis "seed N" row label
TICK_FONTSIZE  = 8
CAPTION_FS     = 14   # "(a) GP" captions
CAPTION_Y      = -0.35

# ════════════════════════════════════════════════════════════════════════════
# DISCOVERY + LOADERS
# ════════════════════════════════════════════════════════════════════════════
def _raw_task_dir(task):
    return RESULTS_DIR / "raw" / task

def _seeds_in_dir(method_dir: Path):
    if not method_dir.exists():
        return set()
    seeds = set()
    for p in method_dir.glob(f"{MODEL}/seed_*.pkl"):
        m = re.match(r"seed_(\d+)\.pkl", p.name)
        if m:
            seeds.add(int(m.group(1)))
    return seeds

def _method_dir_name(opt, lam):
    return online_opt_name(lam) if opt == "online" else opt

def discover_seeds(lam):
    d = _raw_task_dir(TASK_NAME)
    needed = sorted(set(PCA_FIT_METHODS) | set(METHODS))
    sets = []
    for opt in needed:
        s_ = _seeds_in_dir(d / _method_dir_name(opt, lam))
        if s_:
            sets.append(s_)
    return sorted(set.intersection(*sets)) if sets and len(sets) == len(needed) else []

def load_baseline_one(opt, seed):
    try:
        if store.exists(TASK_NAME, opt, MODEL, seed):
            return store.load(TASK_NAME, opt, MODEL, seed)
    except Exception:
        pass
    return None

def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA, arms=ONLINE_ARMS_SUFFIX):
    name = f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"
    if arms:
        name += f"_arms_{arms}"
    return name

def load_method_one(opt, seed, lam):
    if opt == "online":
        name = online_opt_name(lam)
        exists = False
        try:
            exists = store.exists(TASK_NAME, name, MODEL, seed)
        except Exception as e:
            print(f"[online DEBUG] {TASK_NAME} seed={seed} lambda={lam}: "
                  f"exists() raised {e!r}")
        if exists:
            return store.load(TASK_NAME, name, MODEL, seed)
        print(f"[online DEBUG] {TASK_NAME} seed={seed} lambda={lam}: "
              f"'{name}' / {MODEL} not found")
        return None
    return load_baseline_one(opt, seed)

def make_task():
    from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
    return OnlineExperimentRunner._make_task(TASK_NAME)

# ════════════════════════════════════════════════════════════════════════════
# PREP: fit PCA + build background grid, for ONE seed at ONE lambda
# (this becomes one ROW of the final figure)
# ════════════════════════════════════════════════════════════════════════════
def prepare_row(seed, lam, task):
    bounds = np.asarray(task.bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    all_needed = list(dict.fromkeys(PCA_FIT_METHODS + METHODS))
    loaded = {}
    for opt in all_needed:
        r = load_method_one(opt, seed, lam)
        if r is not None:
            loaded[opt] = r

    print(f"[row DEBUG] {TASK_NAME} seed={seed} lambda={lam}: loaded methods = "
          f"{list(loaded.keys())}  (fit-needs {PCA_FIT_METHODS}, plot-needs {METHODS})")

    baselines = {opt: loaded[opt] for opt in METHODS if opt in loaded}
    if not baselines:
        print(f"[skip] no plottable method data for {TASK_NAME}, seed={seed}, lambda={lam}")
        return None

    fit_methods_avail = [opt for opt in PCA_FIT_METHODS if opt in loaded]
    if not fit_methods_avail:
        print(f"[skip] no PCA_FIT_METHODS data for {TASK_NAME}, seed={seed}, lambda={lam} "
              f"(wanted {PCA_FIT_METHODS})")
        return None
    if set(fit_methods_avail) != set(PCA_FIT_METHODS):
        missing = set(PCA_FIT_METHODS) - set(fit_methods_avail)
        print(f"[warn] {TASK_NAME} seed={seed} lambda={lam}: missing PCA_FIT_METHODS "
              f"{missing}; fitting PCA on available subset {fit_methods_avail}")

    fit_pts = np.vstack([np.asarray(loaded[o].xs, dtype=float) for o in fit_methods_avail])
    pca = PCA(n_components=2).fit(fit_pts)
    evr = pca.explained_variance_ratio_
    print(f"{TASK_NAME} seed={seed}: PCA fit on {fit_methods_avail} -- "
          f"PC1={evr[0]:.2%}  PC2={evr[1]:.2%}  total={evr.sum():.2%}")

    plot_pts = np.vstack([np.asarray(baselines[o].xs, dtype=float) for o in baselines])
    extent_pts = np.vstack([fit_pts, plot_pts])
    proj = pca.transform(extent_pts)
    p1min, p1max = proj[:, 0].min(), proj[:, 0].max()
    p2min, p2max = proj[:, 1].min(), proj[:, 1].max()
    r1, r2 = p1max - p1min, p2max - p2min
    g1 = np.linspace(p1min - PCA_PAD * r1, p1max + PCA_PAD * r1, GRID_N)
    g2 = np.linspace(p2min - PCA_PAD * r2, p2max + PCA_PAD * r2, GRID_N)
    G1, G2 = np.meshgrid(g1, g2)

    grid_hi = np.clip(pca.inverse_transform(np.column_stack([G1.ravel(), G2.ravel()])),
                      lo, hi)
    Z = np.array([task.evaluate(p) for p in grid_hi]).reshape(GRID_N, GRID_N)

    opt_2d = None
    try:
        x_opt = getattr(task, "x_star", None)
        if callable(x_opt):
            x_opt = x_opt()
        if x_opt is not None:
            opt_2d = pca.transform(np.asarray(x_opt, dtype=float).reshape(1, -1))[0]
    except Exception:
        opt_2d = None

    return dict(pca=pca, baselines=baselines, G1=G1, G2=G2, Z=Z,
                g1=g1, g2=g2, opt_2d=opt_2d)

# ════════════════════════════════════════════════════════════════════════════
# ONE FIGURE per lambda: rows = seeds, columns = METHODS
# ════════════════════════════════════════════════════════════════════════════
def fig_seed_rows(lam):
    task = make_task()
    avail_seeds = discover_seeds(lam)

    seeds_to_use = [s for s in SEEDS if (not avail_seeds or s in avail_seeds)]
    if not seeds_to_use:
        print(f"[abort] lambda={lam}: no usable seeds")
        return
    if avail_seeds and len(seeds_to_use) < len(SEEDS):
        missing = sorted(set(SEEDS) - set(seeds_to_use))
        print(f"[warn] lambda={lam}: seeds {missing} unavailable, plotting {seeds_to_use}")

    rows = []
    for seed in seeds_to_use:
        row = prepare_row(seed, lam, task)
        if row is not None:
            row["seed"] = seed
            rows.append(row)

    if not rows:
        print(f"[abort] lambda={lam}: nothing plottable across all seeds")
        return

    nrows, ncols = len(rows), len(METHODS)
    # fig, axes = plt.subplots(nrows, ncols,
    #                         #  figsize=(PANEL_W * ncols, PANEL_H * nrows),
    #                         figsize=(16, 20),
    #                          squeeze=False)
    fig, axes = plt.subplots(10, 4, figsize=(20, 24))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)  # increase hspace until captions clear the xlabel

    k = 0
    for i, row in enumerate(rows):
        pca = row["pca"]
        for j, opt in enumerate(METHODS):
            ax = axes[i][j]

            ax.contourf(row["G1"], row["G2"], row["Z"], levels=30,
                        cmap="viridis", alpha=0.85)
            o = row["opt_2d"]
            if o is not None and row["g1"][0] <= o[0] <= row["g1"][-1] \
                    and row["g2"][0] <= o[1] <= row["g2"][-1]:
                ax.scatter(o[0], o[1], marker="*", s=200, color="red",
                           edgecolors="black", linewidths=1, zorder=6)

            res = row["baselines"].get(opt)
            if res is not None:
                p = pca.transform(np.asarray(res.xs, dtype=float))
                color = OPT_COLORS.get(opt, "gray")
                ax.plot(p[:, 0], p[:, 1], color=color, alpha=0.4, lw=1, zorder=3)
                ax.scatter(p[:, 0], p[:, 1], color=color, s=40,
                           edgecolors="white", linewidths=0.8, zorder=5)
                if ANNOTATE:
                    for n, (x, y) in enumerate(p, start=1):
                        ax.annotate(str(n), (x, y), textcoords="offset points",
                                    xytext=(3, 3), fontsize=5, color=color,
                                    fontweight="bold", zorder=8,
                                    path_effects=[pe.withStroke(linewidth=1.2,
                                                                foreground="white")])
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="white")

            ax.set_xlabel("PC 1", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"seed {row['seed']}", fontsize=LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

            ax.text(0.5, CAPTION_Y, f"({string.ascii_lowercase[k % 26]}) "
                                    f"{OPT_LABELS.get(opt, opt)}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=CAPTION_FS)

            if i == 0:
                color = OPT_COLORS.get(opt, "gray")
                traj_handle = mlines.Line2D(
                    [], [], color=color, marker="o", markersize=7,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.8, linewidth=1.3, alpha=0.8,
                    label=OPT_LABELS.get(opt, opt),
                )
                opt_handle = mlines.Line2D(
                    [], [], marker="*", color="none", markersize=12,
                    markerfacecolor="red", markeredgecolor="black",
                    linestyle="None", label="Global optimum",
                )
                ax.legend(
                    handles=[traj_handle, opt_handle],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=2,
                    fontsize=9,
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )
            k += 1

    fig.suptitle(f"{TASK_LABEL}   (lambda = {lam:.2f})", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    outdir = FIG_ROOT / "main" / "grid_olympus_suzuki_seed_rows"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / (
        f"trajectory_pca_{TASK_NAME}_lambda{lam:.2f}_alpha_{MAIN_ALPHA:.2f}_eta_{MAIN_ETA:.2f}"
    )
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.pdf / .png  @ {DPI} dpi")


# ════════════════════════════════════════════════════════════════════════════
# SWEEP: one figure per lambda
# ════════════════════════════════════════════════════════════════════════════
def sweep_all():
    for lam in LAMBDA_LIST:
        print(f"\n{'='*80}\n[sweep] {TASK_NAME}  lambda={lam}  (rows = {len(SEEDS)} seeds)\n{'='*80}")
        fig_seed_rows(lam)


if __name__ == "__main__":
    sweep_all()