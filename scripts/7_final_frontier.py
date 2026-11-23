#!/usr/bin/env python3
"""
Pareto regret–cost trade-off figure for the online-switching hybrid BO paper.

One point per λ:
    x = mean cumulative LLM cost (USD)          [bootstrap CI on the mean]
    y = median final simple regret              [bootstrap CI on the median]
Lower-left is better.

VISUAL DESIGN
-------------
EVERY configuration is colored by λ on a continuous colormap — nothing is
rendered in dead gray. Pareto membership is encoded by marker *weight* rather
than by hue, so the λ signal survives on all points:

    Pareto-optimal : large filled disc, white ring, full opacity, drop shadow
    dominated      : smaller hollow ring in the same λ color, faded

EVERY node carries its own `λ = …` label, placed by a collision-aware search
over candidate offsets (with leader lines when a label must be pushed away).

Same read path as generate_figures.py:
    results/raw/<task>/online_hybrid_lambda_<λ>_alpha_<a>_eta_<e>/<model>/seed_*.pkl
Same output directory:
    results/figures/main/<task>/fig5_frontier.pdf   (vector)
    results/figures/main/<task>/fig5_frontier.png   (400 dpi preview)

Run from the project root (2_HybridBO/hybrid_bo):
    python scripts/final_frontier.py
"""
import os
import re
import sys
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
matplotlib.rcParams["savefig.dpi"] = 600
matplotlib.rcParams["axes.titleweight"] = "bold"
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ── Project root & imports ──────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parent.parent      # .../2_HybridBO/hybrid_bo
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

# ── Static config (must match generate_figures.py) ──────────────────────────
MODEL      = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7
BASELINES  = ["gp", "llm", "llambo", "lgbo"]

# ── The full task set, in reading order for the grid figure ─────────────────
# NOTE: these strings must match the directory names under results/raw/.
# `olympus_colors_bob` is a guess at the "colors_bob" task's on-disk name —
# if the folder is called something else, fix it here.
GRID_TASKS = [
    "hartmann6",
    "bayesmark_SVM_breast",
    "yahpo_lcbench_7593",
    "lasso_breast_cancer",
    "schwefel",
    "bayesmark_SVM_digits",
    "lasso_diabetes",
    "olympus_suzuki",
]

TASK_TITLES = {
    "lasso_breast_cancer": "Lasso Breast",
    "yahpo_lcbench_7593": "LCBench 7593",
    "hartmann6": "Hartmann6",
    "schwefel": "Schwefel",
    "bayesmark_SVM_digits": "SVM Digits",
    "bayesmark_SVM_breast": "SVM Breast",
    "lasso_diabetes": "Lasso Diabetes",
    "olympus_suzuki": "Suzuki",
    "bayesmark_RF_breast": "RF Breast",
}

N_BOOT   = 10_000        # bootstrap resamples for the CIs
CI_LEVEL = 0.95          # two-sided confidence level
RNG_SEED = 0             # bootstrap reproducibility (does not touch the data)

# ── Visual knobs ────────────────────────────────────────────────────────────
CMAP_NAME      = "viridis"   # perceptually uniform, colorblind-safe, print-safe
SHOW_KNEE      = True        # star + callout on the recommended operating point
LABEL_EVERY    = True        # label EVERY node with its λ
LABEL_FONTSIZE = 8.5
S_PARETO       = 130         # marker area, Pareto points
S_DOMINATED    = 58          # marker area, dominated points
ALPHA_DOM      = 0.55        # opacity of dominated markers

# Frontier rendering.
#   STAIRCASE      : orthogonal step boundary (the textbook Pareto frontier).
#                    Set False for a plain diagonal polyline through the points.
#   SHOW_GUIDE     : faint dashed diagonal through the Pareto points, drawn under
#                    the staircase — the dotted "trend" line in the classic figure.
#   SHADE_DOMINATED: tint the region dominated by the frontier (up-and-right).
STAIRCASE       = True
SHOW_GUIDE      = True
SHADE_DOMINATED = True
STAIR_COLOR     = "#B03A2E"  # the red step line in the reference figure
STAIR_EXTEND    = 0.06       # how far the open arms run, as a fraction of range

_ONLINE_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)$"
)

# ════════════════════════════════════════════════════════════════════════════
# Discovery  (identical semantics to generate_figures.py)
# ════════════════════════════════════════════════════════════════════════════
def _raw_task_dir(task):
    return RESULTS_DIR / "raw" / task

def discover_main_lambdas(task, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    d = _raw_task_dir(task)
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

def _seeds_in_dir(method_dir: Path):
    if not method_dir.exists():
        return set()
    seeds = set()
    for p in method_dir.glob(f"{MODEL}/seed_*.pkl"):
        m = re.match(r"seed_(\d+)\.pkl", p.name)
        if m:
            seeds.add(int(m.group(1)))
    return seeds

def discover_seeds_intersection(task, lam, alpha=MAIN_ALPHA, eta=MAIN_ETA,
                                include_baselines=True):
    d = _raw_task_dir(task)
    online_name = f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"
    sets = [_seeds_in_dir(d / online_name)]
    if include_baselines:
        for opt in BASELINES:
            sets.append(_seeds_in_dir(d / opt))
    common = set.intersection(*sets) if sets and all(sets) else set()
    return sorted(common)

# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════
def online_opt_name(lam, alpha, eta):
    return f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"

def load_online(task, lam, alpha, eta, seeds):
    name = online_opt_name(lam, alpha, eta)
    out = []
    for s in seeds:
        try:
            if store.exists(task, name, MODEL, s):
                out.append(store.load(task, name, MODEL, s))
        except Exception:
            pass
    return out

def load_baseline(task, opt, seeds):
    out = []
    for s in seeds:
        try:
            if store.exists(task, opt, MODEL, s):
                out.append(store.load(task, opt, MODEL, s))
        except Exception:
            pass
    return out

# ════════════════════════════════════════════════════════════════════════════
# Regret reference  (f* from the task, else best-found on disk)
# ════════════════════════════════════════════════════════════════════════════
def task_fstar(task):
    try:
        from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
        t = OnlineExperimentRunner._make_task(task)
        fs = t.f_star()
        return None if fs is None else float(fs)
    except Exception:
        return None

_FSTAR_CACHE = {}
def fstar_for(task, all_seeds):
    if task in _FSTAR_CACHE:
        return _FSTAR_CACHE[task]
    fs = task_fstar(task)
    if fs is None:
        best = -np.inf
        for lam in discover_main_lambdas(task):
            for r in load_online(task, lam, MAIN_ALPHA, MAIN_ETA, all_seeds):
                best = max(best, r.f_best)
        for opt in BASELINES:
            for r in load_baseline(task, opt, all_seeds):
                best = max(best, r.f_best)
        fs = best if np.isfinite(best) else None
    _FSTAR_CACHE[task] = fs
    return fs

def regret(f_best_value, task, all_seeds):
    fs = fstar_for(task, all_seeds)
    if fs is None:
        return np.nan
    r = fs - float(f_best_value)
    return 0.0 if r < 0 else r          # f* is an upper bound; negatives are fp noise

# ════════════════════════════════════════════════════════════════════════════
# Bootstrap CIs
# ════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(x, stat=np.nanmedian, n_boot=N_BOOT, level=CI_LEVEL, rng=None):
    """Return (point_estimate, lo, hi). Degrades gracefully for n < 2."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    point = float(stat(x))
    if x.size < 2:
        return point, point, point
    rng = np.random.default_rng(RNG_SEED) if rng is None else rng
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)])
    return point, float(lo), float(hi)

# ════════════════════════════════════════════════════════════════════════════
# Pareto front  (automatic; minimize BOTH cost and regret)
# ════════════════════════════════════════════════════════════════════════════
def pareto_mask(cost, regret_):
    """
    Boolean mask of nondominated points for simultaneous minimization.
    Point i is DOMINATED iff some j is <= on both objectives and < on at least
    one. Ties on both objectives are mutually nondominated (both kept).
    """
    c = np.asarray(cost, dtype=float)
    r = np.asarray(regret_, dtype=float)
    n = c.size
    nondom = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(c[i]) or not np.isfinite(r[i]):
            nondom[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            le_both = (c[j] <= c[i]) and (r[j] <= r[i])
            lt_one  = (c[j] <  c[i]) or  (r[j] <  r[i])
            if le_both and lt_one:
                nondom[i] = False
                break
    return nondom

def knee_point(cost, regret_, mask):
    """
    Recommended operating point = knee of the Pareto front: the nondominated
    point closest to the utopia corner (min cost, min regret) after min-max
    normalizing both axes over the Pareto set, so the choice is scale-free.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    if idx.size == 1:
        return int(idx[0])
    c = np.asarray(cost, dtype=float)[idx]
    r = np.asarray(regret_, dtype=float)[idx]
    c_rng = float(np.ptp(c)) or 1.0
    r_rng = float(np.ptp(r)) or 1.0
    cn = (c - c.min()) / c_rng
    rn = (r - r.min()) / r_rng
    d = np.hypot(cn, rn)
    return int(idx[int(np.argmin(d))])

# ════════════════════════════════════════════════════════════════════════════
# Staircase (step) attainment frontier
# ════════════════════════════════════════════════════════════════════════════
def pareto_staircase(cost, regret_, mask, extend=0.06):
    """
    Build the STEP (staircase) Pareto frontier — the standard textbook form.

    A diagonal line between consecutive nondominated points is a lie: it implies
    achievable trade-offs on the segment that were never measured. The honest
    boundary is orthogonal. For minimize-minimize with cost on x and regret on y,
    each nondominated point dominates the quadrant up-and-right of it, so the
    attainment boundary steps:

        from point k (sorted by ASCENDING cost, hence DESCENDING regret),
        go RIGHT to the next point's cost at the CURRENT regret,
        then DOWN to the next point's regret.

    This traces the lower-left envelope: every location on or below-left of the
    staircase is unattainable by the measured configurations, everything above
    and right is dominated.

    The two open ends are extended by `extend` (a fraction of each axis range) so
    the frontier reads as a boundary rather than a floating polyline, matching the
    usual textbook rendering.

    Returns (xs, ys) ready to hand to ax.plot / ax.fill_between.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([]), np.array([])

    c = np.asarray(cost, dtype=float)[idx]
    r = np.asarray(regret_, dtype=float)[idx]
    o = np.argsort(c)                     # ascending cost
    c, r = c[o], r[o]

    xs, ys = [], []

    # leading arm: straight up from the cheapest point (nothing cheaper exists)
    span_y = float(np.ptp(regret_)) or max(abs(float(r[0])), 1.0)
    xs.append(c[0]); ys.append(r[0] + extend * span_y)

    # the steps themselves
    for k in range(c.size):
        xs.append(c[k]); ys.append(r[k])                # arrive at point k
        if k + 1 < c.size:
            xs.append(c[k + 1]); ys.append(r[k])        # RIGHT at current regret
            # (the DOWN leg is implicit: next iteration appends (c[k+1], r[k+1]))

    # trailing arm: straight right from the lowest-regret point
    span_x = float(np.ptp(cost)) or max(abs(float(c[-1])), 1.0)
    xs.append(c[-1] + extend * span_x); ys.append(r[-1])

    return np.asarray(xs), np.asarray(ys)


# ════════════════════════════════════════════════════════════════════════════
# Style
# ════════════════════════════════════════════════════════════════════════════
def set_academic_style():
    plt.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       400,
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,     # TrueType — editable/embeddable vector text
        "ps.fonttype":       42,
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset":  "dejavuserif",
        "font.size":         11,
        "axes.titlesize":    13.5,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   9,
        "axes.linewidth":    0.9,
        "axes.edgecolor":    "0.30",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "0.88",
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",
        "grid.alpha":        0.9,
        "axes.axisbelow":    True,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.color":       "0.30",
        "ytick.color":       "0.30",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon":    False,
    })

# ════════════════════════════════════════════════════════════════════════════
# Data assembly — one record per λ, original data preserved exactly
# ════════════════════════════════════════════════════════════════════════════
def collect_points(task, all_seeds):
    rng = np.random.default_rng(RNG_SEED)
    lams, cost_mu, cost_lo, cost_hi = [], [], [], []
    reg_md, reg_lo, reg_hi, n_seeds = [], [], [], []

    for lam in discover_main_lambdas(task):
        seeds = discover_seeds_intersection(task, lam)
        if not seeds:
            seeds = discover_seeds_intersection(task, lam, include_baselines=False)
        runs = load_online(task, lam, MAIN_ALPHA, MAIN_ETA, seeds)
        if not runs:
            continue

        costs = np.array([r.total_cost for r in runs], dtype=float)
        regs  = np.array([regret(r.f_best, task, all_seeds) for r in runs],
                         dtype=float)

        cm_, cl_, ch_ = bootstrap_ci(costs, stat=np.nanmean,   rng=rng)
        rm_, rl_, rh_ = bootstrap_ci(regs,  stat=np.nanmedian, rng=rng)

        lams.append(float(lam))
        cost_mu.append(cm_); cost_lo.append(cl_); cost_hi.append(ch_)
        reg_md.append(rm_);  reg_lo.append(rl_);  reg_hi.append(rh_)
        n_seeds.append(len(runs))

    return dict(lam=np.array(lams),
                cost=np.array(cost_mu),
                cost_lo=np.array(cost_lo), cost_hi=np.array(cost_hi),
                regret=np.array(reg_md),
                reg_lo=np.array(reg_lo), reg_hi=np.array(reg_hi),
                n=np.array(n_seeds))

# ════════════════════════════════════════════════════════════════════════════
# Label placement — EVERY node gets a λ tag
# ════════════════════════════════════════════════════════════════════════════
def place_all_labels(ax, fig, xs, ys, lam, mask, colors):
    """
    Label every node with `λ = …`.

    Strategy: for each point, try a ring of candidate offsets (near first, then
    progressively farther). Accept the first that collides with neither an
    already-placed label nor any data marker. When a label ends up far from its
    node, draw a hairline leader so the association stays unambiguous.

    Labels are drawn in the node's own λ color (darkened for legibility) and
    given a white halo so they stay readable over grid lines and error bars.
    Pareto labels are bold; dominated labels are regular weight.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # keep-out boxes around the markers themselves
    marker_boxes = []
    for x, y, m in zip(xs, ys, mask):
        px, py = ax.transData.transform((x, y))
        pad = 11 if m else 8
        marker_boxes.append(
            matplotlib.transforms.Bbox.from_bounds(px - pad, py - pad,
                                                   2 * pad, 2 * pad))

    # candidate offsets, ordered near -> far, all 8 compass directions.
    # Vertical slots come first: a λ sweep tends to spread along the cost axis,
    # so above/below a node is usually the emptiest real estate.
    rings = [10, 15, 21, 29, 38]
    dirs  = [(0, 1), (0, -1),
             (1, 0.75), (-1, 0.75), (1, -0.95), (-1, -0.95),
             (1.25, 0), (-1.25, 0)]
    candidates = [(dx * r, dy * r) for r in rings for dx, dy in dirs]

    LEADER_MIN = 22          # px: beyond this the tie to the node needs a leader

    def _mk(i, dx, dy):
        ha = "left" if dx > 1e-6 else ("right" if dx < -1e-6 else "center")
        va = "bottom" if dy > 1e-6 else ("top" if dy < -1e-6 else "center")
        return ax.annotate(
            rf"$\lambda={lam[i]:g}$", (xs[i], ys[i]),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va=va,
            fontsize=LABEL_FONTSIZE,
            color=tuple(0.62 * np.array(colors[i][:3])),   # darken for contrast
            fontweight="bold" if mask[i] else "normal",
            zorder=11, annotation_clip=False,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )

    placed = []
    for i in np.argsort(~mask):          # Pareto points get first pick
        chosen = None
        for dx, dy in candidates:
            t = _mk(i, dx, dy)
            bb = t.get_window_extent(renderer=renderer).expanded(1.10, 1.20)
            if (any(bb.overlaps(p) for p in placed) or
                    any(bb.overlaps(m) for m in marker_boxes)):
                t.remove()
                continue
            chosen = (t, bb, dx, dy)
            break

        if chosen is None:               # fully crowded: take the farthest slot
            dx, dy = candidates[-1]
            t = _mk(i, dx, dy)
            chosen = (t, t.get_window_extent(renderer=renderer), dx, dy)

        t, bb, dx, dy = chosen
        placed.append(bb)

        # Leader line for labels that had to be pushed away. The annotation's
        # own arrowprops is the only reliable anchor here: it knows the text box
        # in offset-point space, so the line lands on the label edge rather than
        # on a stale data-coordinate guess.
        if np.hypot(dx, dy) > LEADER_MIN:
            t.set_annotation_clip(False)
            t.arrow_patch = None
            t.arrowprops = dict(
                arrowstyle="-", lw=0.55,
                color=tuple(0.62 * np.array(colors[i][:3])),
                alpha=0.5, shrinkA=1.5, shrinkB=6.5,
                connectionstyle="arc3,rad=0.0",
            )
            # re-instantiate so matplotlib actually builds the arrow patch
            t.remove()
            t2 = ax.annotate(
                rf"$\lambda={lam[i]:g}$", (xs[i], ys[i]),
                textcoords="offset points", xytext=(dx, dy),
                ha=t.get_ha(), va=t.get_va(),
                fontsize=LABEL_FONTSIZE,
                color=tuple(0.62 * np.array(colors[i][:3])),
                fontweight="bold" if mask[i] else "normal",
                zorder=11, annotation_clip=False,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                arrowprops=dict(arrowstyle="-", lw=0.55,
                                color=tuple(0.62 * np.array(colors[i][:3])),
                                alpha=0.5, shrinkA=1.5, shrinkB=6.5),
            )
            placed[-1] = t2.get_window_extent(renderer=renderer).expanded(1.10, 1.20)

# ════════════════════════════════════════════════════════════════════════════
# Panel drawing — one task onto a supplied axis
# ════════════════════════════════════════════════════════════════════════════
def draw_panel(ax, fig, task, all_seeds, norm, cmap,
               show_xlabel=True, show_ylabel=True, panel_tag=None):
    """
    Render one task's Pareto trade-off onto `ax`.

    `norm` is SHARED across all panels: λ is normalized over the UNION of λ
    values found in every task, so a given color means the same λ everywhere in
    the grid. Normalizing per-panel would make identical colors mean different
    λ in different subplots — a subtle but real way to mislead a reader.

    Returns a small summary dict (or None when the task has no results).
    """
    D = collect_points(task, all_seeds)
    if D["lam"].size == 0:
        ax.text(0.5, 0.5, "no results", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.55")
        ax.set_title(TASK_TITLES.get(task, task), pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        return None

    lam, cost, reg = D["lam"], D["cost"], D["regret"]
    xerr = np.clip(np.vstack([cost - D["cost_lo"], D["cost_hi"] - cost]), 0, None)
    yerr = np.clip(np.vstack([reg  - D["reg_lo"],  D["reg_hi"]  - reg ]), 0, None)

    mask = pareto_mask(cost, reg)
    knee = knee_point(cost, reg, mask) if SHOW_KNEE else None

    colors = [cmap(norm(l)) for l in lam]     # SHARED norm -> comparable hues

    dom  = ~mask
    pidx = np.flatnonzero(mask)
    pidx = pidx[np.argsort(cost[pidx])]       # order the front by cost

    # ── error bars: tinted per-point, drawn under the markers ────────────────
    for i in range(lam.size):
        ax.errorbar(cost[i], reg[i],
                    xerr=xerr[:, i:i+1], yerr=yerr[:, i:i+1],
                    fmt="none",
                    ecolor=colors[i],
                    elinewidth=1.0 if mask[i] else 0.7,
                    capsize=2.2 if mask[i] else 1.5,
                    capthick=1.0 if mask[i] else 0.7,
                    alpha=0.85 if mask[i] else 0.40,
                    zorder=3)

    # ── Pareto frontier (orthogonal staircase) ───────────────────────────────
    if pidx.size:
        if STAIRCASE:
            sx, sy = pareto_staircase(cost, reg, mask, extend=STAIR_EXTEND)
            if SHADE_DOMINATED and sx.size:
                top = float(np.nanmax(reg + yerr[1])) + 0.5 * (float(np.ptp(reg)) or 1.0)
                ax.fill_between(sx, sy, top, color=STAIR_COLOR, alpha=0.05,
                                lw=0, zorder=1)
            if SHOW_GUIDE and pidx.size > 1:
                ax.plot(cost[pidx], reg[pidx], ls=(0, (3, 3)), lw=0.85,
                        color=STAIR_COLOR, alpha=0.45, zorder=4)
            ax.plot(sx, sy, "-", lw=1.4, color=STAIR_COLOR, alpha=0.95,
                    zorder=5, solid_capstyle="butt", solid_joinstyle="miter")
        else:
            ax.plot(cost[pidx], reg[pidx], "-", lw=4.0, color="0.55",
                    alpha=0.18, zorder=4, solid_capstyle="round")
            ax.plot(cost[pidx], reg[pidx], "-", lw=1.2, color="0.32",
                    alpha=0.95, zorder=5, solid_capstyle="round")

    # ── dominated: hollow rings, SAME λ color, faded ─────────────────────────
    if dom.any():
        ax.scatter(cost[dom], reg[dom],
                   s=S_DOMINATED, marker="o",
                   facecolors="white",
                   edgecolors=[colors[i] for i in np.flatnonzero(dom)],
                   linewidths=1.4, alpha=ALPHA_DOM, zorder=6)

    # ── Pareto: filled discs, white ring, subtle shadow ──────────────────────
    ax.scatter(cost[mask], reg[mask],
               s=S_PARETO, marker="o",
               c=lam[mask], cmap=cmap, norm=norm,
               edgecolors="white", linewidths=1.2, zorder=8,
               path_effects=[pe.withSimplePatchShadow(
                   offset=(1.0, -1.0), alpha=0.16, shadow_rgbFace="black")])

    # ── recommended operating point ──────────────────────────────────────────
    if knee is not None:
        ax.scatter([cost[knee]], [reg[knee]], s=340, marker="*",
                   facecolors="none", edgecolors=STAIR_COLOR,
                   linewidths=1.5, zorder=10)

    # ── axes cosmetics ───────────────────────────────────────────────────────
    title = TASK_TITLES.get(task, task)
    if panel_tag:
        title = f"({panel_tag}) {title}"
    ax.set_title(title, pad=8, fontsize=12)
    if show_xlabel:
        ax.set_xlabel("Cumulative LLM cost")
    if show_ylabel:
        ax.set_ylabel("Regret")
    ax.margins(x=0.20, y=0.22)          # breathing room for the λ tags

    # ── λ tag on EVERY node ──────────────────────────────────────────────────
    if LABEL_EVERY:
        place_all_labels(ax, fig, cost, reg, lam, mask, colors)

    return dict(task=task, lam=lam, mask=mask, cost=cost, reg=reg, knee=knee)


# ════════════════════════════════════════════════════════════════════════════
# The grid figure — N panels, one shared legend, one shared colorbar
# ════════════════════════════════════════════════════════════════════════════
def _panel_tags(n):
    """'a', 'b', ..., 'z', then 'aa', 'ab', ... if we ever need more than 26."""
    import string
    letters = string.ascii_lowercase
    out = []
    for i in range(n):
        if i < 26:
            out.append(letters[i])
        else:
            out.append(letters[i // 26 - 1] + letters[i % 26])
    return out

def grid_figure(tasks, seeds_by_task, outname="fig5_frontier_grid", n_cols=2):
    """
    N-panel grid of Pareto trade-offs, one axis per task.

    Layout: row-major reading order, `n_cols` columns (default 2), as many rows
    as needed for len(tasks). Axis labels appear only on the outer edges (x on
    the bottom row of each column, y on the left column) so the grid stays
    uncluttered; each panel keeps its OWN x/y limits, because cost and regret
    scales differ by orders of magnitude across tasks and forcing a shared
    scale would flatten most panels into an unreadable smear.

    ONE legend and ONE colorbar are shared for the whole figure. Any unused
    trailing axes (when len(tasks) doesn't fill the last row) are hidden.
    """
    # ── shared λ normalization across every panel ────────────────────────────
    all_lams = []
    for t in tasks:
        all_lams.extend(discover_main_lambdas(t))
    if not all_lams:
        print("[skip] grid: no λ found for any task.")
        return
    lam_min, lam_max = float(min(all_lams)), float(max(all_lams))
    norm = Normalize(vmin=lam_min, vmax=lam_max)
    cmap = plt.get_cmap(CMAP_NAME)

    n = len(tasks)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.2 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    axes = axes.ravel()
    tags = _panel_tags(n)

    summaries = []
    for k, task in enumerate(tasks):
        row, col = divmod(k, n_cols)
        # last physical row that actually holds a panel, per column
        is_last_row_in_col = (k + n_cols) >= n
        s = draw_panel(
            axes[k], fig, task, seeds_by_task.get(task, []),
            norm=norm, cmap=cmap,
            show_xlabel=is_last_row_in_col,
            show_ylabel=(col == 0),
            panel_tag=tags[k],
        )
        summaries.append(s)

    # hide any unused trailing axes (grid doesn't divide evenly)
    for k in range(n, len(axes)):
        axes[k].axis("off")

    # ── ONE shared colorbar for λ, spanning the full grid ────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:n].tolist(), pad=0.020, fraction=0.030,
                      aspect=34)
    cb.set_label(r"cost penalty $\lambda$", fontsize=12, labelpad=9)
    cb.outline.set_linewidth(0.7)
    cb.outline.set_edgecolor("0.45")
    cb.ax.tick_params(labelsize=9.5, length=3, color="0.45")
    uniq = sorted(set(all_lams))
    cb.set_ticks(np.linspace(lam_min, lam_max, min(6, len(uniq))))

    # ── ONE shared legend, below the grid ────────────────────────────────────
    handles = [
        Line2D([], [], ls="none", marker="o", ms=9,
               mfc="#3F7F93", mec="white", mew=1.3, label="Pareto-optimal (solid dot)"),
        Line2D([], [], ls="none", marker="o", ms=7,
               mfc="white", mec="#3F7F93", mew=1.5, alpha=0.85,
               label="dominated"),
        Line2D([], [], color=STAIR_COLOR, lw=1.6, label="Pareto frontier"),
    ]
    if SHOW_KNEE:
        handles.append(Line2D([], [], ls="none", marker="*", ms=15,
                              mfc="none", mec=STAIR_COLOR, mew=1.5,
                              label="recommended"))
    # fig.legend(handles=handles, loc="lower center", ncol=len(handles),
    #            bbox_to_anchor=(0.46, -0.005), frameon=False,
    #            handletextpad=0.6, columnspacing=2.0, fontsize=11)
    fig.legend(handles=handles, loc="upper right", ncol=1,
           bbox_to_anchor=(0.985, 0.985), frameon=True,
           framealpha=0.9, edgecolor="0.75",
           handletextpad=0.6, columnspacing=2.0, fontsize=11)

    fig.subplots_adjust(left=0.075, right=0.90, top=0.955, bottom=0.09,
                        wspace=0.24, hspace=0.34)

    # ── save: vector PDF + 400-dpi PNG preview ───────────────────────────────
    outdir = FIG_ROOT / "main" / "_grid"
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{outname}.pdf"
    png = outdir / f"{outname}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    for s in summaries:
        if s is None:
            continue
        print(f"  [{s['task']}] Pareto λ: {[f'{l:g}' for l in s['lam'][s['mask']]]}"
              f"   dominated λ: {[f'{l:g}' for l in s['lam'][~s['mask']]]}")
        if s["knee"] is not None:
            k = s["knee"]
            print(f"      recommended (knee): λ={s['lam'][k]:g}  "
                  f"cost=${s['cost'][k]:.5f}  regret={s['reg'][k]:.5f}")
    print(f"  saved: {pdf.relative_to(PKG_ROOT)}")
    print(f"  saved: {png.relative_to(PKG_ROOT)}")

def single_figure(task, seeds, outname=None):
    """One task, one axis, shared-style Pareto panel."""
    lambdas = discover_main_lambdas(task)
    if not lambdas:
        print(f"[skip] {task}: no λ on disk.")
        return

    norm = Normalize(vmin=float(min(lambdas)), vmax=float(max(lambdas)))
    cmap = plt.get_cmap(CMAP_NAME)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    s = draw_panel(ax, fig, task, seeds, norm=norm, cmap=cmap,
                   show_xlabel=True, show_ylabel=True, panel_tag=None)

    # colorbar for λ
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046, aspect=26)
    cb.set_label(r"cost penalty $\lambda$", fontsize=12, labelpad=9)
    cb.outline.set_linewidth(0.7)
    cb.outline.set_edgecolor("0.45")
    cb.ax.tick_params(labelsize=9.5, length=3, color="0.45")
    uniq = sorted(set(lambdas))
    cb.set_ticks(np.linspace(norm.vmin, norm.vmax, min(6, len(uniq))))

    handles = [
        Line2D([], [], ls="none", marker="o", ms=9,
               mfc="#3F7F93", mec="white", mew=1.3, label="Pareto-optimal (solid)"),
        Line2D([], [], ls="none", marker="o", ms=7,
               mfc="white", mec="#3F7F93", mew=1.5, alpha=0.85, label="dominated (hollow)"),
        Line2D([], [], color=STAIR_COLOR, lw=1.6, label="Pareto frontier"),
    ]
    if SHOW_KNEE:
        handles.append(Line2D([], [], ls="none", marker="*", ms=15,
                              mfc="none", mec=STAIR_COLOR, mew=1.5,
                              label="recommended"))
        
    ax.legend(handles=handles, loc="best", frameon=False,
              handletextpad=0.6, fontsize=10)
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handletextpad=0.6, fontsize=10)
    # fig.legend(handles=handles, loc="upper right", ncol=1,
    #        bbox_to_anchor=(0.985, 0.985), frameon=True,
    #        framealpha=0.9, edgecolor="0.75",
    #     handletextpad=0.6, columnspacing=2.0, fontsize=11)

    outname = outname or f"fig5_frontier_{task}"
    outdir = FIG_ROOT / "main" / task
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{outname}.pdf"
    png = outdir / f"{outname}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")   # high pixel
    plt.close(fig)

    if s is not None:
        print(f"  [{task}] Pareto λ: {[f'{l:g}' for l in s['lam'][s['mask']]]}"
              f"   dominated λ: {[f'{l:g}' for l in s['lam'][~s['mask']]]}")
        if s["knee"] is not None:
            k = s["knee"]
            print(f"      recommended (knee): λ={s['lam'][k]:g}  "
                  f"cost=${s['cost'][k]:.5f}  regret={s['reg'][k]:.5f}")
    print(f"  saved: {pdf.relative_to(PKG_ROOT)}")
    print(f"  saved: {png.relative_to(PKG_ROOT)}")


def _seeds_for_task(task):
    """Union of intersected seeds across every λ discovered for this task."""
    lambdas = discover_main_lambdas(task)
    if not lambdas:
        return []
    seed_sets = [
        set(discover_seeds_intersection(task, lam, include_baselines=False))
        for lam in lambdas
    ]
    return sorted(set().union(*seed_sets)) or list(range(0, 10))


def main():
    set_academic_style()
    print(f"Project root: {PKG_ROOT}")
    print(f"Figures → {FIG_ROOT}\n")

    seeds_by_task = {}

    # ── one single-panel figure per task in GRID_TASKS ───────────────────────
    for task in GRID_TASKS:
        lambdas = discover_main_lambdas(task)
        if not lambdas:
            print(f"[abort] {task}: no online results on disk.")
            continue

        seeds = _seeds_for_task(task)
        seeds_by_task[task] = seeds
        print(f"=== {task}  λ∈{lambdas}  seeds={seeds} ===")
        single_figure(task, seeds)
        print()

    # ── one combined grid figure across every task that had data ────────────
    tasks_with_data = [t for t in GRID_TASKS if seeds_by_task.get(t)]
    if tasks_with_data:
        print(f"=== grid figure: {tasks_with_data} ===")
        grid_figure(tasks_with_data, seeds_by_task)
    else:
        print("[skip] grid: no task had any results.")

    print("\nDONE.")
# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    main()