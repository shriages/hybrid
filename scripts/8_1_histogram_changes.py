#!/usr/bin/env python3
"""
Single-task, multi-lambda stacked figure -- PROBABILITY TIER ONLY.

Three subplots total, one per lambda (top -> bottom), each showing the
arm-probability curves P_j(a) vs batch j for that lambda:

    row 1 (top)    lambda = LAMBDAS[0]
    row 2 (middle) lambda = LAMBDAS[1]
    row 3 (bottom) lambda = LAMBDAS[2]

This is the timeline/histogram-free version of 8_1_lambda_stack_histogram.py --
just tier (a), "what the policy BELIEVED", repeated for each lambda.

Everything you'd normally want to change lives in the CONFIG block:
    TASK     -- which task directory to read
    LAMBDAS  -- the three lambda values, top -> bottom
    SEEDS    -- which seeds to include (None -> every seed on disk)

Run from project root (2_HybridBO/hybrid_bo):
    python scripts/8_2_lambda_prob_stack.py
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
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

PKG_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

# ══════════════════════════════════════════════════════════════════════════
# CONFIG  -- the knobs you asked for
# ══════════════════════════════════════════════════════════════════════════
TASK    = "lasso_diabetes"          # task key on disk: results/raw/<TASK>/...
TITLE   = "Lasso / Diabetes"        # display name (None -> reuse TASK)

LAMBDAS = [0.5, 0.7, 1.0]           # <<< top, middle, bottom
ROW_LABEL = ["top", "middle", "bottom"]

SEEDS   = None                      # <<< SPECIFY SEEDS HERE. None -> all on disk.
                                     #     e.g. [0], [0, 1, 2], list(range(10))
# ──────────────────────────────────────────────────────────────────────────

FIG_WIDTH  = 5.0
FIG_HEIGHT = 1.9 * len(LAMBDAS)     # one probability-tier subplot per lambda
PROB_SHADE = True                   # +/-1 std band on P_j(a)
SHOW_TITLE = True                   # set False for a caption-only paper figure

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
# DISCOVERY + LOADERS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════
def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    return f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"


def discover_lambdas(task, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    d = RESULTS_DIR / "raw" / task
    if not d.exists():
        return []
    lams = set()
    for sub in d.iterdir():
        m = _ONLINE_RE.match(sub.name)
        if not m:
            continue
        if abs(float(m["alpha"]) - alpha) < 1e-9 and abs(float(m["eta"]) - eta) < 1e-9:
            lams.add(round(float(m["lam"]), 4))
    return sorted(lams)


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
# ONE ROW  -- probability tier only, for a single (task, lambda)
# ══════════════════════════════════════════════════════════════════════════
def draw_prob_row(fig, ax, task, title, lam, row_label):
    want, missing = resolve_seeds(task, lam)
    if missing:
        print(f"[warn] λ={lam}: seeds not on disk, ignored: {missing}")
    if not want:
        ax.text(0.5, 0.5, f"{title}  λ={lam}\n(no usable seeds)",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_axis_off()
        print(f"[abort row] {task} λ={lam}: no usable seeds "
              f"(on disk: {discover_seeds(task, lam)})")
        return

    runs, seeds = load_online(task, lam, want)
    if not runs:
        ax.text(0.5, 0.5, f"{title}  λ={lam}\n(no runs)", ha="center",
                va="center", fontsize=8, transform=ax.transAxes)
        ax.set_axis_off()
        print(f"[abort row] {task} λ={lam}: nothing loaded")
        return

    print(f"=== {row_label}: {title}  (task={task}, λ={lam}, seeds={seeds}) ===")

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
    ax.set_xlim(0, J - 1)
    ax.set_ylabel("$P_j(a)$", fontsize=7, labelpad=2)
    ax.set_xlabel("batch $j$", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if SHOW_TITLE:
        ax.set_title(f"{title}  (λ={lam:g}, {len(runs)} seeds)",
                      fontsize=8.5, pad=4)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE  -- one task, 3 stacked probability subplots (top / middle / bottom)
# ══════════════════════════════════════════════════════════════════════════
def main():
    task  = TASK
    title = TITLE or TASK

    assert len(LAMBDAS) == 3, "LAMBDAS must have exactly 3 entries (top/mid/bottom)"

    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT),
                             sharex=False)
    fig.subplots_adjust(left=0.13, right=0.975, top=0.90, bottom=0.08,
                        hspace=0.65)

    for idx, lam in enumerate(LAMBDAS):
        draw_prob_row(fig, axes[idx], task, title, lam, ROW_LABEL[idx])

    fig.legend(handles=[Patch(facecolor=ARM_COLOR[a], label=a) for a in ARMS],
               ncol=len(ARMS), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               frameon=False, fontsize=9, handlelength=1.4,
               columnspacing=1.8, handletextpad=0.5)

    # ── output: filed under the task, tagged with the lambdas ──────────────
    outdir = FIG_ROOT / "main" / task
    outdir.mkdir(parents=True, exist_ok=True)
    lam_tag = "_".join(f"{l:g}" for l in LAMBDAS)
    stem = outdir / f"lambda_prob_stack_{lam_tag}"
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()