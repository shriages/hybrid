#!/usr/bin/env python3
"""
PCA trajectory grid: rows = tasks, columns = plotted methods.

Within each row/task:
  - The PCA rotation (2D basis) is FIT on the trajectories of
    PCA_FIT_METHODS (e.g. GP, LLAMBO, LGBO) -- this defines what "PC 1" /
    "PC 2" mean for that task.
  - The PANELS actually drawn (one per column) are METHODS (e.g. GP, LGBO,
    Online) -- their trajectories are TRANSFORMED (not re-fit) into that
    same PCA basis, so everything lives in one consistent 2D space per row.

Optionally (SHARE_PCA_ACROSS_ROWS), the very first row's fitted PCA can
also be reused (transform-only) for every subsequent row, so all rows
share one global rotation too. Off by default -- each row fits its own
PCA from PCA_FIT_METHODS.

ITERATION: the whole grid is regenerated once per (seed, lambda) pair, for
seed in SEEDS and lambda in LAMBDA_LIST. Each (seed, lambda) combo writes
its own .pdf / .png so nothing gets overwritten. seed applies to every
loaded method (gp/llambo/lgbo/online); lambda only affects which "online"
run directory is used.

Panels are captioned (a) GP, (b) LGBO, ... continuing across rows, with
the task name as the row's shared y-axis label.

Run from the project root:
    python 2_PCA_trajectory_grid.py
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
# One entry per ROW: (task_dir_name, pretty y-axis label). Seed is no longer
# fixed per-task here -- it's supplied by the SEEDS loop below and applied
# uniformly across all rows/tasks for a given figure.
TASKS = [
    # this is for paper
    ("hartmann6", "Hartmann6"),
    ("schwefel",  "Schwefel"),
    # this is my detection
    # ("olympus_suzuki", "Olympus Suzuki"),
]

# Methods used ONLY to define the PCA rotation for each row (not plotted
# unless they also appear in METHODS below).
PCA_FIT_METHODS = ["gp", "llambo", "lgbo"]

# Methods actually PLOTTED as columns (transformed into the PCA basis
# defined by PCA_FIT_METHODS above -- not re-fit).
# "online" is NOT a baseline dir: it is the ORIGINAL switching run with the
# full 4-arm pool (gp, llm, llambo, lgbo), stored WITHOUT an _arms_ suffix.
METHODS    = ["gp", "lgbo", "online"]
ONLINE_ARMS_SUFFIX = None        # None -> original 4-arm pool (no suffix)

# MODEL      = "deepseek-v4-flash"
MODEL = "openai-5.4-mini"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7

# ── ITERATION: seeds and lambdas to sweep for the "online" method ──────────
SEEDS       = list(range(10))               # 0..9
LAMBDA_LIST = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0]        # edit to the lambdas you have on disk

OPT_COLORS = {"gp": "#5F5E5A", "llm": "#378ADD", "llambo": "#1D9E75", "lgbo": "#D85A30", "online": "#C0392B"}
OPT_LABELS = {"gp": "GP", "llm": "LLM", "llambo": "LLAMBO", "lgbo": "LGBO", "online": "HybridBO"}

ANNOTATE   = False
GRID_N     = 200
PCA_PAD    = 0.08

# If True, the FIRST row's fitted PCA (from PCA_FIT_METHODS) is reused
# (transform-only) for every subsequent row too, so ALL rows share one
# global rotation. If False (default), each row fits its own PCA from its
# own PCA_FIT_METHODS data. Requires same raw input dimensionality across
# tasks if enabled.
SHARE_PCA_ACROSS_ROWS = False

# ── figure quality / typography ────────────────────────────────────────────
DPI            = 600       # high pixel density
PANEL_W        = 6.0       # inches per panel
PANEL_H        = 5.0
LABEL_FONTSIZE = 30        # x/y axis LABEL size (not the tick numbers)
TICK_FONTSIZE  = 14        # tick numbers
CAPTION_FS     = 30        # "(a) GP" captions
CAPTION_Y      = -0.26     # caption offset below each panel (axes fraction)

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

def discover_baseline_seeds(task, lam):
    """Seeds common to every method we need to LOAD -- the union of
    PCA_FIT_METHODS (for fitting) and METHODS (for plotting), at a given
    online lambda."""
    d = _raw_task_dir(task)
    needed = sorted(set(PCA_FIT_METHODS) | set(METHODS))
    sets = []
    for opt in needed:
        s_ = _seeds_in_dir(d / _method_dir_name(opt, lam))
        if s_:
            sets.append(s_)
    return sorted(set.intersection(*sets)) if sets and len(sets) == len(needed) else []

def load_baseline_one(task, opt, seed):
    try:
        if store.exists(task, opt, MODEL, seed):
            return store.load(task, opt, MODEL, seed)
    except Exception:
        pass
    return None

def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA, arms=ONLINE_ARMS_SUFFIX):
    name = f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"
    if arms:
        name += f"_arms_{arms}"
    return name

def load_method_one(task, opt, seed, lam):
    """Load one method's run. Baselines load from their own dir; 'online'
    loads the switching run from the gp-lgbo pool at the given lambda. Both
    expose .xs."""
    if opt == "online":
        name = online_opt_name(lam)
        exists = False
        try:
            exists = store.exists(task, name, MODEL, seed)
        except Exception as e:
            print(f"[online DEBUG] {task} seed={seed} lambda={lam}: "
                  f"exists() raised {e!r}")
        print(f"[online DEBUG] {task} seed={seed} lambda={lam}: looking for "
              f"'{name}' / {MODEL}  -> exists={exists}")
        if exists:
            r = store.load(task, name, MODEL, seed)
            print(f"[online DEBUG]   loaded, len(xs)={len(getattr(r,'xs',[]))}")
            return r
        return None
    return load_baseline_one(task, opt, seed)

def make_task(task_name):
    from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
    return OnlineExperimentRunner._make_task(task_name)

# ════════════════════════════════════════════════════════════════════════════
# PER-ROW PREP: fit PCA (or reuse one), build background grid
# ════════════════════════════════════════════════════════════════════════════
def prepare_row(task_name, seed, lam, shared_pca=None):
    """Returns dict with pca, grid, Z, per-method results -- or None.

    PCA basis:
      - if shared_pca is provided, it's used to TRANSFORM this row's points
        (no new fit), reusing an earlier row's rotation.
      - otherwise, a fresh PCA is fit on THIS row's PCA_FIT_METHODS data
        (e.g. GP, LLAMBO, LGBO), independent of what gets plotted.

    Panels plotted:
      - always come from METHODS (e.g. GP, LGBO, Online), transformed into
        whichever PCA basis was chosen above. The "online" method is loaded
        at the given lambda.
    """
    task = make_task(task_name)
    bounds = np.asarray(task.bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    # Load everything we might need: union of fit methods and plot methods.
    all_needed = list(dict.fromkeys(PCA_FIT_METHODS + METHODS))  # dedup, keep order
    loaded = {}
    for opt in all_needed:
        r = load_method_one(task_name, opt, seed, lam)
        if r is not None:
            loaded[opt] = r

    print(f"[row DEBUG] {task_name} seed={seed} lambda={lam}: loaded methods = "
          f"{list(loaded.keys())}  (fit-needs {PCA_FIT_METHODS}, plot-needs {METHODS})")

    # Methods used for plotting (columns) -- only those we actually have.
    baselines = {opt: loaded[opt] for opt in METHODS if opt in loaded}
    if not baselines:
        print(f"[skip] no plottable method data for {task_name}, seed={seed}, lambda={lam}")
        return None

    # Methods used to fit the PCA basis -- only those we actually have.
    fit_methods_avail = [opt for opt in PCA_FIT_METHODS if opt in loaded]
    if not fit_methods_avail:
        print(f"[skip] no PCA_FIT_METHODS data for {task_name}, seed={seed}, lambda={lam} "
              f"(wanted {PCA_FIT_METHODS})")
        return None
    if set(fit_methods_avail) != set(PCA_FIT_METHODS):
        missing = set(PCA_FIT_METHODS) - set(fit_methods_avail)
        print(f"[warn] {task_name} seed={seed} lambda={lam}: missing PCA_FIT_METHODS "
              f"{missing}; fitting PCA on available subset {fit_methods_avail}")

    fit_pts = np.vstack([np.asarray(loaded[o].xs, dtype=float) for o in fit_methods_avail])

    if shared_pca is not None:
        pca = shared_pca
        evr = pca.explained_variance_ratio_
        print(f"{task_name}: using SHARED PCA rotation from first row "
              f"(PC1={evr[0]:.2%}  PC2={evr[1]:.2%}  total={evr.sum():.2%})")
    else:
        pca = PCA(n_components=2).fit(fit_pts)
        evr = pca.explained_variance_ratio_
        print(f"{task_name}: PCA fit on {fit_methods_avail} -- "
              f"PC1={evr[0]:.2%}  PC2={evr[1]:.2%}  total={evr.sum():.2%}")

    # Grid extents: based on the union of everything we'll actually draw
    # (fit-basis data + plotted-panel data), so nothing gets clipped out.
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
                g1=g1, g2=g2, opt_2d=opt_2d, evr=evr)

# ════════════════════════════════════════════════════════════════════════════
# ONE FIGURE for a given (seed, lambda)
# ════════════════════════════════════════════════════════════════════════════
def fig_trajectory_grid(seed, lam):
    rows = []
    shared_pca = None  # will hold the FIRST row's fitted PCA, if enabled

    for task_name, label in TASKS:
        avail_seeds = discover_baseline_seeds(task_name, lam)
        if avail_seeds and seed not in avail_seeds:
            print(f"[skip] {task_name}: seed={seed} not available for lambda={lam} "
                  f"(available: {avail_seeds})")
            continue
        if not avail_seeds:
            print(f"[warn] {task_name}: could not verify common seeds for lambda={lam}; "
                  f"attempting seed={seed} anyway")

        pca_to_use = shared_pca if SHARE_PCA_ACROSS_ROWS else None
        row = prepare_row(task_name, seed, lam, shared_pca=pca_to_use)
        if row is not None:
            row.update(task_name=task_name, label=label, seed=seed, lam=lam)
            rows.append(row)
            if SHARE_PCA_ACROSS_ROWS and shared_pca is None:
                shared_pca = row["pca"]   # lock in the rotation from row 1

    if not rows:
        print(f"[abort] nothing to plot for seed={seed}, lambda={lam}")
        return

    nrows, ncols = len(rows), len(METHODS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PANEL_W * ncols, PANEL_H * nrows),
                             squeeze=False)

    k = 0
    for i, row in enumerate(rows):
        pca = row["pca"]
        for j, opt in enumerate(METHODS):
            ax = axes[i][j]

            # background contour
            ax.contourf(row["G1"], row["G2"], row["Z"], levels=30,
                        cmap="viridis", alpha=0.85)
            o = row["opt_2d"]
            if o is not None and row["g1"][0] <= o[0] <= row["g1"][-1] \
                    and row["g2"][0] <= o[1] <= row["g2"][-1]:
                ax.scatter(o[0], o[1], marker="*", s=300, color="red",
                           edgecolors="black", linewidths=1, zorder=6)

            # trajectory
            res = row["baselines"].get(opt)
            if res is not None:
                p = pca.transform(np.asarray(res.xs, dtype=float))
                color = OPT_COLORS.get(opt, "gray")
                ax.plot(p[:, 0], p[:, 1], color=color, alpha=0.4, lw=1, zorder=3)
                ax.scatter(p[:, 0], p[:, 1], color=color, s=55,
                           edgecolors="white", linewidths=0.9, zorder=5)
                if ANNOTATE:
                    for n, (x, y) in enumerate(p, start=1):
                        ax.annotate(str(n), (x, y), textcoords="offset points",
                                    xytext=(4, 4), fontsize=6, color=color,
                                    fontweight="bold", zorder=8,
                                    path_effects=[pe.withStroke(linewidth=1.5,
                                                                foreground="white")])

            # labels: task name on y (left column only), PC 1 on x
            ax.set_xlabel("PC 1", fontsize=16)
            if j == 0:
                ax.set_ylabel(row["label"], fontsize=LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

            # "(a) GP" caption below the panel
            ax.text(0.5, CAPTION_Y, f"({string.ascii_lowercase[k]}) "
                                    f"{OPT_LABELS.get(opt, opt)}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=CAPTION_FS)
            if i == 0:
                color = OPT_COLORS.get(opt, "gray")
                traj_handle = mlines.Line2D(
                    [], [], color=color, marker="o", markersize=8,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.9, linewidth=1.5, alpha=0.8,
                    label=OPT_LABELS.get(opt, opt),
                )
                opt_handle = mlines.Line2D(
                    [], [], marker="*", color="none", markersize=14,
                    markerfacecolor="red", markeredgecolor="black",
                    linestyle="None", label="Global optimum",
                )
                ax.legend(
                    handles=[traj_handle, opt_handle],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=2,
                    fontsize=11,
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )
            k += 1

    # fig.suptitle(f"seed={seed}   lambda={lam:.2f}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    outdir = FIG_ROOT / "main" / "grid" / (
        f"lambda_{lam:.2f}_alpha_{MAIN_ALPHA:.2f}_eta_{MAIN_ETA:.2f}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(r["task_name"] for r in rows)
    stem = outdir / f"trajectory_pca_grid_{tag}_seed{seed}_lambda{lam:.2f}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.pdf / .png  @ {DPI} dpi")


# ════════════════════════════════════════════════════════════════════════════
# SWEEP: seed × lambda
# ════════════════════════════════════════════════════════════════════════════
def sweep_all():
    for lam in LAMBDA_LIST:
        for seed in SEEDS:
            print(f"\n{'='*80}\n[sweep] seed={seed}  lambda={lam}\n{'='*80}")
            fig_trajectory_grid(seed, lam)


if __name__ == "__main__":
    sweep_all()