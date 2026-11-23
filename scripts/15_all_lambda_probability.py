#!/usr/bin/env python3
"""
Multi-lambda x multi-task arm-probability grid -- ONE combined figure.

Layout: 3 stacked BLOCKS, one block per lambda in LAMBDAS (top -> bottom).
Each block is a 2x4 grid (BLOCK_NROWS x BLOCK_NCOLS) covering all 8 tasks in
GRID_TASKS, so the whole figure is effectively 6 rows x 4 cols of panels,
visually grouped into 3 labeled bands ("lambda = 0.5" / "lambda = 1.0" /
"lambda = 1.5") with extra vertical breathing room between bands.

Each individual panel shows the arm-probability curves P_j(a) vs batch j for
ONE (task, lambda) pair -- since lambda is now the block's identity rather
than something overlaid within a panel, color encodes arm only (no linestyle
juggling needed, unlike the earlier "3 lambdas overlaid in one panel"
version).

Everything you'd normally want to change lives in the CONFIG block:
    GRID_TASKS -- the 8 task keys + display titles (one column position each)
    LAMBDAS    -- the three lambda values, top block -> bottom block
    SEEDS      -- which seeds to include (None -> every seed on disk)

Run from project root (2_HybridBO/hybrid_bo):
    python scripts/8_2_lambda_prob_grid_stacked.py
"""
import os, re, sys
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

PKG_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

# ══════════════════════════════════════════════════════════════════════════
# CONFIG -- the knobs you asked for
# ══════════════════════════════════════════════════════════════════════════
GRID_TASKS = [
    dict(task="hartmann6",            title="Hartmann6"),
    dict(task="schwefel",             title="Schwefel"),
    dict(task="yahpo_lcbench_7593",   title="LCBench 7593"),
    dict(task="olympus_suzuki",       title="Suzuki"),
    dict(task="bayesmark_SVM_digits", title="SVM Digits"),
    dict(task="bayesmark_SVM_breast", title="SVM Breast"),
    dict(task="lasso_diabetes",       title="Lasso Diabetes"),
    dict(task="lasso_breast_cancer",  title="Lasso Breast"),
]
BLOCK_NROWS = 2      # rows PER lambda-block
BLOCK_NCOLS = 4      # cols PER lambda-block (2x4 = 8 tasks per block)

LAMBDAS = [0.5, 1.0, 1.5]   # <<< one block each, TOP -> BOTTOM
SEEDS   = None              # <<< SPECIFY SEEDS HERE. None -> all on disk.
                             #     e.g. [0], [0, 1, 2], list(range(10))
# ──────────────────────────────────────────────────────────────────────────

PANEL_W = 4.2
PANEL_H = 2.4
BLOCK_VSPACE   = 0.10   # extra gridspec gap between lambda-blocks (fig fraction)
WITHIN_HSPACE  = 0.55   # hspace within a block (rows share the block's title area)
WITHIN_WSPACE  = 0.35

PROB_SHADE = True       # +/-1 std band per arm
SHOW_PANEL_TITLE = True # per-panel task-name(+seed count) title

MODEL      = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7
ARMS       = ["gp", "llm", "llambo", "lgbo"]
PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
ARM_COLOR  = {a: PALETTE[i] for i, a in enumerate(ARMS)}

RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

_ONLINE_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)$"
)


# ══════════════════════════════════════════════════════════════════════════
# DISCOVERY + LOADERS (unchanged from the single-task script)
# ══════════════════════════════════════════════════════════════════════════
def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    return f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"


def discover_seeds(task, lam):
    d = RESULTS_DIR / "raw" / task / online_opt_name(lam) / MODEL
    if not d.exists():
        return []
    seeds = []
    for p in d.glob("seed_*.pkl"):
        m = re.match(r"seed_(\d+)\.pkl", p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load_online(task, lam, seeds):
    name = online_opt_name(lam)
    runs, kept = [], []
    for s in seeds:
        try:
            if store.exists(task, name, MODEL, s):
                runs.append(store.load(task, name, MODEL, s))
                kept.append(s)
        except Exception:
            pass
    return runs, kept


def resolve_seeds(task, lam):
    """Apply the SEEDS config against what's actually on disk."""
    on_disk = discover_seeds(task, lam)
    if not on_disk:
        return [], []
    if SEEDS is None:
        return on_disk, []
    want = [s for s in SEEDS if s in on_disk]
    missing = [s for s in SEEDS if s not in on_disk]
    return want, missing


# ══════════════════════════════════════════════════════════════════════════
# ONE PANEL -- probability tier for a single (task, lambda)
# ══════════════════════════════════════════════════════════════════════════
def draw_panel(ax, task, title, lam):
    want, missing = resolve_seeds(task, lam)
    if missing:
        print(f"[warn] {task} λ={lam}: seeds not on disk, ignored: {missing}")
    if not want:
        ax.text(0.5, 0.5, f"{title}\n(no usable seeds)", ha="center",
                va="center", fontsize=7, transform=ax.transAxes)
        ax.set_axis_off()
        print(f"[abort panel] {task} λ={lam}: no usable seeds "
              f"(on disk: {discover_seeds(task, lam)})")
        return

    runs, seeds = load_online(task, lam, want)
    if not runs:
        ax.text(0.5, 0.5, f"{title}\n(no runs)", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        ax.set_axis_off()
        print(f"[abort panel] {task} λ={lam}: nothing loaded")
        return

    print(f"    {title} (task={task}, λ={lam}, {len(runs)} seeds)")

    mats = [r.prob_matrix for r in runs]
    J = min(m.shape[0] for m in mats)
    P = np.stack([m[:J] for m in mats], axis=0)
    mean, std = P.mean(axis=0), P.std(axis=0)
    batches = np.arange(J)
    for a, name in enumerate(ARMS):
        ax.step(batches, mean[:, a], where="post", color=ARM_COLOR[name],
                linewidth=1.2)
        if PROB_SHADE and len(runs) >= 2:
            ax.fill_between(batches, mean[:, a] - std[:, a],
                            mean[:, a] + std[:, a], step="post",
                            alpha=0.18, color=ARM_COLOR[name], linewidth=0)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(J - 1, 1))
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if SHOW_PANEL_TITLE:
        ax.set_title(f"{title} ({len(runs)} seeds)", fontsize=7.5, pad=3)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE -- 3 stacked 2x4 blocks (one per lambda), ONE shared legend
# ══════════════════════════════════════════════════════════════════════════
def main():
    n_blocks = len(LAMBDAS)
    n_panels = len(GRID_TASKS)
    n_slots = BLOCK_NROWS * BLOCK_NCOLS
    if n_panels > n_slots:
        print(f"[warn] {n_panels} tasks but each block only has {n_slots} "
              f"slots; extra tasks will be dropped per block.")

    fig_h = PANEL_H * BLOCK_NROWS * n_blocks + 0.6 * n_blocks  # rough, tight_layout fixes it
    fig = plt.figure(figsize=(PANEL_W * BLOCK_NCOLS, fig_h))

    outer_gs = gridspec.GridSpec(
        n_blocks, 1, figure=fig,
        top=0.93, bottom=0.05, left=0.06, right=0.99,
        hspace=BLOCK_VSPACE,
    )

    for b, lam in enumerate(LAMBDAS):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            BLOCK_NROWS, BLOCK_NCOLS, subplot_spec=outer_gs[b],
            hspace=WITHIN_HSPACE, wspace=WITHIN_WSPACE,
        )

        print(f"\n=== block {b}: λ={lam} ===")
        for idx, spec in enumerate(GRID_TASKS):
            if idx >= n_slots:
                break
            row, col = divmod(idx, BLOCK_NCOLS)
            ax = fig.add_subplot(inner_gs[row, col])
            draw_panel(ax, spec["task"], spec["title"], lam)

            is_bottom = (row == BLOCK_NROWS - 1) or (idx + BLOCK_NCOLS >= n_panels)
            is_left = (col == 0)
            if is_bottom:
                ax.set_xlabel("batch $j$", fontsize=7, labelpad=1)
            if is_left:
                ax.set_ylabel("$P_j(a)$", fontsize=7, labelpad=2)

        # block label ("λ = 0.5"), placed just above this block's own area
        pos = outer_gs[b].get_position(fig)
        fig.text(0.5, pos.y1 + 0.006, rf"$\lambda = {lam:g}$",
                 ha="center", va="bottom", fontsize=13, fontweight="bold")

    # ── ONE shared legend for arm colors, above the whole figure ────────────
    arm_handles = [Patch(facecolor=ARM_COLOR[a], label=a) for a in ARMS]
    fig.legend(handles=arm_handles, ncol=len(ARMS), loc="upper center",
              bbox_to_anchor=(0.5, 0.998), frameon=False, fontsize=10,
              handlelength=1.4, columnspacing=1.8, handletextpad=0.5)

    outdir = FIG_ROOT / "main" / "_grid"
    outdir.mkdir(parents=True, exist_ok=True)
    lam_tag = "_".join(f"{l:g}" for l in LAMBDAS)
    stem = outdir / f"lambda_prob_grid_stacked_{lam_tag}"
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()