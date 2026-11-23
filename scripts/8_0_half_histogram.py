#!/usr/bin/env python3
"""
Single-task, single-figure version of the three-tier story (no 2x2 subplot grid).

One task, three stacked tiers:
    (a) arm probabilities P_j(a) vs batch j   -- what the policy BELIEVED
    (b) per-seed timeline of chosen arms       -- what it actually RAN
    (c) half-trajectory histograms             -- what that SUMS TO

Everything you'd normally want to change lives in the CONFIG block:
    TASK    -- which task directory to read
    LAMBDA  -- which lambda to use (None -> auto-pick per PICK_LAMBDA)
    SEEDS   -- which seeds to include (None -> every seed on disk)

Tier (a) averages P_j(a) over the selected seeds; tier (b) draws one row per
selected seed; tier (c) sums iterations over the selected seeds. So SEEDS
controls all three tiers at once.

Run from project root (2_HybridBO/hybrid_bo):
    python scripts/single_task_story.py
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
# CONFIG  -- the three knobs you asked for
# ══════════════════════════════════════════════════════════════════════════
TASK   = "olympus_suzuki"          # task key on disk: results/raw/<TASK>/...
TITLE  = "Olympus Suzuki"          # display name (None -> reuse TASK)

LAMBDA = 1.5                  # <<< SPECIFY LAMBDA HERE.  None -> auto-pick.
SEEDS  = list(range(10))      # <<< SPECIFY SEEDS HERE.   None -> all on disk.
                              #     e.g. [0], [0, 1, 2], list(range(10))

PICK_LAMBDA = "median"        # "median" | "first" | "last"  (only when LAMBDA=None)
# ──────────────────────────────────────────────────────────────────────────

HALVES = [("$t$: 2–31",  2, 31),
          ("$t$: 32–61", 32, 61)]
NORMALIZE  = True
FIG_WIDTH  = 5.0              # single-column-ish; one task only now
FIG_HEIGHT = 6.4
HEIGHT_RATIO = [1.25, 1.0, 1.0]   # probs : timeline : histograms
PROB_SHADE = True             # +/-1 std band on P_j(a)
SHOW_SEED_LABELS = True       # with one task there IS room for seed ticks
SHOW_TITLE = True             # set False for a caption-only paper figure

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


def pick_lambda(task):
    lams = discover_lambdas(task)
    if not lams:
        return None
    if PICK_LAMBDA == "first":
        return lams[0]
    if PICK_LAMBDA == "last":
        return lams[-1]
    return lams[len(lams) // 2]


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


def per_iteration_arms(r):
    labels = []
    for j, arm_idx in enumerate(r.batch_arms):
        labels.extend([r.arm_names[arm_idx]] * int(r.batch_schedule[j]))
    return np.array(labels, dtype=object)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE  -- one task, three tiers, no outer grid
# ══════════════════════════════════════════════════════════════════════════
def main():
    task  = TASK
    title = TITLE or TASK

    lam = LAMBDA if LAMBDA is not None else pick_lambda(task)
    if lam is None:
        print(f"[abort] {task}: no λ on disk "
              f"(found: {discover_lambdas(task)})")
        return

    want, missing = resolve_seeds(task, lam)
    if missing:
        print(f"[warn] seeds not on disk, ignored: {missing}")
    if not want:
        print(f"[abort] {task} λ={lam}: no usable seeds "
              f"(on disk: {discover_seeds(task, lam)})")
        return

    runs, seeds = load_online(task, lam, want)
    if not runs:
        print(f"[abort] {task} λ={lam}: nothing loaded")
        return

    print(f"=== {title}  (task={task}, λ={lam}, seeds={seeds}) ===")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=HEIGHT_RATIO,
                           hspace=0.25, wspace=0.12,
                           left=0.135, right=0.975, top=0.92, bottom=0.075)

    split_t = HALVES[0][2]

    # ── (a) arm probabilities P_j(a) ───────────────────────────────────────
    ax_p = fig.add_subplot(gs[0, :])
    mats = [r.prob_matrix for r in runs]
    J = min(m.shape[0] for m in mats)
    P = np.stack([m[:J] for m in mats], axis=0)
    mean, std = P.mean(axis=0), P.std(axis=0)
    batches = np.arange(J)
    for a, name in enumerate(ARMS):
        ax_p.step(batches, mean[:, a], where="post", color=ARM_COLOR[name],
                  linewidth=1.2)
        if PROB_SHADE and len(runs) >= 2:
            ax_p.fill_between(batches, mean[:, a] - std[:, a],
                              mean[:, a] + std[:, a], step="post",
                              alpha=0.18, color=ARM_COLOR[name], linewidth=0)
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_xlim(0, J - 1)
    ax_p.set_ylabel("$P_j(a)$", fontsize=8, labelpad=2)
    ax_p.set_xlabel("batch $j$", fontsize=8, labelpad=1)
    ax_p.tick_params(labelsize=7, length=2, pad=1)
    ax_p.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax_p.spines[sp].set_visible(False)
    if SHOW_TITLE:
        ax_p.set_title(f"{title}  ({len(runs)} seeds)", fontsize=9.5, pad=1)

    # ── (b) per-seed timeline ──────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, :])
    for row, r in enumerate(runs):
        bounds = list(r.batch_boundaries)
        starts = [0] + bounds[:-1]
        for j, arm_idx in enumerate(r.batch_arms):
            name = r.arm_names[arm_idx]
            ax_t.barh(row, r.batch_schedule[j], left=starts[j], height=0.85,
                      color=ARM_COLOR.get(name), edgecolor="white",
                      linewidth=0.25)
    ax_t.axvline(split_t, color="black", ls="--", lw=0.8, zorder=5)
    if SHOW_SEED_LABELS:
        ax_t.set_yticks(range(len(runs)))
        ax_t.set_yticklabels([str(s) for s in seeds], fontsize=6)
    else:
        ax_t.set_yticks([])
    ax_t.set_ylabel("seed", fontsize=8, labelpad=3)
    ax_t.set_xlabel("iteration $t$", fontsize=8, labelpad=1)
    ax_t.tick_params(axis="x", labelsize=7, length=2, pad=1)
    ax_t.tick_params(axis="y", length=0, pad=1)
    ax_t.set_xlim(0, max(r.batch_boundaries[-1] for r in runs))
    ax_t.set_ylim(-0.6, len(runs) - 0.4)
    for sp in ("top", "right"):
        ax_t.spines[sp].set_visible(False)

    # ── (c) half-trajectory histograms ─────────────────────────────────────
    counts = {lab: {a: 0 for a in ARMS} for lab, _, _ in HALVES}
    for r in runs:
        arms_t = per_iteration_arms(r)
        for lab, lo, hi in HALVES:
            for a in arms_t[lo - 1: hi]:
                if a in counts[lab]:
                    counts[lab][a] += 1

    x = np.arange(len(ARMS))
    axes_h = []
    for k, (lab, lo, hi) in enumerate(HALVES):
        ax = fig.add_subplot(gs[2, k], sharey=axes_h[0] if axes_h else None)
        axes_h.append(ax)
        vals = np.array([counts[lab][a] for a in ARMS], dtype=float)
        if NORMALIZE and vals.sum() > 0:
            vals = vals / vals.sum()
        ax.bar(x, vals, color=[ARM_COLOR[a] for a in ARMS],
               edgecolor="white", linewidth=0.4, width=0.75)
        ax.set_title(lab, fontsize=7.5, pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS, fontsize=7, rotation=45, ha="right")
        ax.grid(alpha=0.25, axis="y", linewidth=0.4)
        ax.tick_params(labelsize=7, length=2, pad=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    axes_h[0].set_ylabel("frac. of iters" if NORMALIZE else "# iters",
                         fontsize=8, labelpad=2)

    # stdout summary
    for lab, _, _ in HALVES:
        row = "  ".join(f"{a}={counts[lab][a]}" for a in ARMS)
        print(f"   {lab}: {row}")

    fig.legend(handles=[Patch(facecolor=ARM_COLOR[a], label=a) for a in ARMS],
               ncol=len(ARMS), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               frameon=False, fontsize=9, handlelength=1.4,
               columnspacing=1.8, handletextpad=0.5)

    # ── output: filed under the λ, tagged with the seeds ───────────────────
    outdir = FIG_ROOT / "main" / task
    outdir.mkdir(parents=True, exist_ok=True)
    seed_tag = ("all" if SEEDS is None
                else f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 3
                else "_".join(map(str, seeds)))
    stem = outdir / f"story_lambda_{lam:.2f}_seeds_{seed_tag}"
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()