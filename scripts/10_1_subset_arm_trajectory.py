#!/usr/bin/env python3
"""
All-8-tasks grid figure: one row per task, 2 columns per row --
  left  = arm probabilities P_j(a) vs batch j   (tier (a) from
          10_subset_arm_figure.py's three-tier story figure)
  right = half-trajectory histogram, grouped bars per arm
          (t:2-31 vs t:32-61 shown as two side-by-side bars per arm,
          tier (c) from the same script, collapsed into one subplot)

Produces exactly 2 figures (one per arm pool):
    results/figures/gp_lgbo/all_tasks/grid_lambda_<λ>_seeds_<tag>.png/.pdf
    results/figures/llm_lgbo/all_tasks/grid_lambda_<λ>_seeds_<tag>.png/.pdf

Reuses the same read-in (ResultStore + directory layout + discovery regex),
CONFIG values, and MASTER_ARM_COLOR palette as 10_subset_arm_figure.py, so
this figure stays visually/data-consistent with the three-tier ones.

Run from project root (2_HybridBO/hybrid_bo):
    python scripts/12_all_tasks_grid_figure.py

────────────────────────────────────────────────────────────────────────────
Two choices I made without asking (change them at the top of CONFIG below
if they're not what you want):
  FIG_LAMBDA = 0.00   -- every row uses this single lambda value.
  Histogram style = grouped bars per arm (t:2-31 solid, t:32-61 hatched),
  rather than two separate mini-subplots, so it fits in one right-column
  panel per row.
────────────────────────────────────────────────────────────────────────────
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
# CONFIG  (identical to 10_subset_arm_figure.py so figures stay comparable)
# ══════════════════════════════════════════════════════════════════════════
TASKS = ["schwefel", "hartmann6", "bayesmark_SVM_breast", "bayesmark_SVM_digits",
          "lasso_diabetes", "lasso_breast_cancer", "yahpo_lcbench_7593", "olympus_suzuki"]

TASK_TITLES = {
    "schwefel":             "Schwefel",
    "hartmann6":             "Hartmann6",
    "bayesmark_SVM_breast":  "SVM / breast",
    "bayesmark_SVM_digits":  "SVM / digits",
    "lasso_diabetes":        "Lasso / diabetes",
    "lasso_breast_cancer":   "Lasso / breast cancer",
    "yahpo_lcbench_7593":    "YAHPO LCBench 7593",
    "olympus_suzuki":        "Olympus Suzuki",
}

FIG_LAMBDA = 10.0   # <<< single lambda used for every row in this grid figure

SEEDS = list(range(10))      # None -> all seeds on disk

HALVES = [("t:2-31",  2, 31),
          ("t:32-61", 32, 61)]
NORMALIZE = True

MODEL      = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7

MASTER_ARM_COLOR = {
    "gp":     "#4C72B0",
    "llm":    "#DD8452",
    "llambo": "#55A868",
    "lgbo":   "#C44E52",
}

ARM_POOLS = [
    {"arms": ["gp", "lgbo"],  "suffix": "gp-lgbo"},
    {"arms": ["llm", "lgbo"], "suffix": "llm-lgbo"},
]

ARMS        = ARM_POOLS[0]["arms"]
ARMS_SUFFIX = ARM_POOLS[0]["suffix"]
ARM_COLOR   = {a: MASTER_ARM_COLOR[a] for a in ARMS}


def configure_arm_pool(pool):
    global ARMS, ARM_COLOR, ARMS_SUFFIX
    ARMS = pool["arms"]
    ARMS_SUFFIX = pool["suffix"]
    ARM_COLOR = {a: MASTER_ARM_COLOR[a] for a in ARMS}


RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

_ONLINE_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)"
    r"(?:_arms_(?P<arms>[a-z0-9\-]+))?$"
)

# ── grid-figure-specific knobs ───────────────────────────────────────────
ROW_HEIGHT   = 1.35          # inches per task row
FIG_WIDTH    = 9.0
PROB_SHADE   = True
HALF_BAR_WIDTH = 0.36         # width of each half's bar within an arm slot

# ══════════════════════════════════════════════════════════════════════════
# DISCOVERY + LOADERS  (unchanged from 10_subset_arm_figure.py)
# ══════════════════════════════════════════════════════════════════════════
def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    name = f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"
    if ARMS_SUFFIX:
        name += f"_arms_{ARMS_SUFFIX}"
    return name


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
    on_disk = discover_seeds(task, lam)
    if not on_disk:
        return [], []
    if SEEDS is None:
        return on_disk, []
    want = [s for s in SEEDS if s in on_disk]
    missing = [s for s in SEEDS if s not in on_disk]
    return want, missing


def per_iteration_arms(r):
    labels = []
    for j, arm_idx in enumerate(r.batch_arms):
        labels.extend([r.arm_names[arm_idx]] * int(r.batch_schedule[j]))
    return np.array(labels, dtype=object)


# ══════════════════════════════════════════════════════════════════════════
# ROW DRAWERS
# ══════════════════════════════════════════════════════════════════════════
def draw_prob_path_row(ax, runs, title, lam, is_top):
    """Left column: arm probabilities P_j(a) vs batch j, averaged over seeds."""
    mats = [r.prob_matrix for r in runs]
    J = min(m.shape[0] for m in mats)
    P = np.stack([m[:J] for m in mats], axis=0)
    mean, std = P.mean(axis=0), P.std(axis=0)
    batches = np.arange(J)
    for a, name in enumerate(ARMS):
        ax.step(batches, mean[:, a], where="post", color=ARM_COLOR[name],
                linewidth=1.1)
        if PROB_SHADE and len(runs) >= 2:
            ax.fill_between(batches, mean[:, a] - std[:, a],
                             mean[:, a] + std[:, a], step="post",
                             alpha=0.18, color=ARM_COLOR[name], linewidth=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, J - 1)
    ax.set_ylabel(f"{title}\n$P_j(a)$", fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if is_top:
        ax.set_title(f"arm probabilities $P_j(a)$  (λ={lam:.2f})",
                     fontsize=9, pad=4)


def draw_histogram_row(ax, runs, lam, is_top):
    """Right column: grouped bar histogram, one bar-pair per arm (half 1 vs half 2)."""
    counts = {lab: {a: 0 for a in ARMS} for lab, _, _ in HALVES}
    for r in runs:
        arms_t = per_iteration_arms(r)
        for lab, lo, hi in HALVES:
            for a in arms_t[lo - 1: hi]:
                if a in counts[lab]:
                    counts[lab][a] += 1

    x = np.arange(len(ARMS))
    n_halves = len(HALVES)
    offsets = (np.arange(n_halves) - (n_halves - 1) / 2) * HALF_BAR_WIDTH
    hatches = ["", "///"]
    for k, (lab, lo, hi) in enumerate(HALVES):
        vals = np.array([counts[lab][a] for a in ARMS], dtype=float)
        if NORMALIZE and vals.sum() > 0:
            vals = vals / vals.sum()
        ax.bar(x + offsets[k], vals, width=HALF_BAR_WIDTH,
               color=[ARM_COLOR[a] for a in ARMS],
               edgecolor="white", linewidth=0.4,
               hatch=hatches[k % len(hatches)])
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS, fontsize=6.5)
    ax.set_ylabel("frac. of iters" if NORMALIZE else "# iters",
                  fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.grid(alpha=0.25, axis="y", linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if is_top:
        ax.set_title("half-trajectory arm usage", fontsize=9, pad=4)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE -- 8 tasks x (prob path | histogram), one arm pool
# ══════════════════════════════════════════════════════════════════════════
def draw_grid(lam):
    n = len(TASKS)
    fig = plt.figure(figsize=(FIG_WIDTH, ROW_HEIGHT * n + 0.6))
    gs = gridspec.GridSpec(n, 2, figure=fig, width_ratios=[1.7, 1.0],
                           hspace=0.55, wspace=0.28,
                           left=0.09, right=0.98, top=0.94, bottom=0.06)

    any_row = False
    for i, task in enumerate(TASKS):
        title = TASK_TITLES.get(task, task)
        want, missing = resolve_seeds(task, lam)
        if missing:
            print(f"[warn] {task} λ={lam}: seeds not on disk, ignored: {missing}")
        if not want:
            print(f"[skip] {task} λ={lam}: no usable seeds "
                  f"(on disk: {discover_seeds(task, lam)})")
            continue
        runs, seeds = load_online(task, lam, want)
        if not runs:
            print(f"[skip] {task} λ={lam}: nothing loaded")
            continue

        print(f"=== {ARMS_SUFFIX}: {title}  (λ={lam}, seeds={seeds}) ===")
        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])
        draw_prob_path_row(ax_left, runs, title, lam, is_top=(i == 0))
        draw_histogram_row(ax_right, runs, lam, is_top=(i == 0))
        if i == n - 1:
            ax_left.set_xlabel("batch $j$", fontsize=7.5, labelpad=2)
        any_row = True

    if not any_row:
        print(f"[skip] {ARMS_SUFFIX}: no task had usable data at λ={lam}")
        plt.close(fig)
        return False

    handles = [Patch(facecolor=ARM_COLOR[a], label=a) for a in ARMS]
    handles += [Patch(facecolor="white", edgecolor="black",
                       hatch=h if h else None, label=lab)
                for (lab, _, _), h in zip(HALVES, ["", "///"])]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=8.5,
               handlelength=1.4, columnspacing=1.6, handletextpad=0.5)

    outdir = FIG_ROOT / ARMS_SUFFIX.replace("-", "_") / "all_tasks"
    outdir.mkdir(parents=True, exist_ok=True)
    seed_tag = "all" if SEEDS is None else f"{SEEDS[0]}-{SEEDS[-1]}" if len(SEEDS) > 3 \
        else "_".join(map(str, SEEDS))
    stem_name = f"grid_lambda_{lam:.2f}_seeds_{seed_tag}"
    for ext in ("pdf", "png"):
        path = outdir / f"{stem_name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)
    return True


def main():
    ok_pools = []
    for pool in ARM_POOLS:
        configure_arm_pool(pool)
        print(f"\n########## arm pool: {ARMS} (suffix={ARMS_SUFFIX}) ##########")
        success = draw_grid(FIG_LAMBDA)
        if success:
            ok_pools.append(ARMS_SUFFIX)

    print(f"\n=== overall: {len(ok_pools)}/{len(ARM_POOLS)} grid figures saved "
          f"({ok_pools}) ===")


if __name__ == "__main__":
    main()