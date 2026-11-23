#!/usr/bin/env python3
"""
Polar "circle" figure of LLM-backend usage across ALL tasks.

Two windows (t = 1-31 and t = 31-61) are drawn as TWO STACKED polar panels in a
single figure. Panels are placed with explicit axes rectangles so they never
overlap, and each panel is identified by a y-label: "(a)  t = 1-31" etc.

Task identity is NOT written around the rim (that wastes a lot of radius).
Instead EVERY TASK GETS ITS OWN GLYPH -- a unique (marker shape, color) pair --
drawn just outside its wedge. The glyph -> task-name mapping lives in a single
tall legend column on the RIGHT of the panels.

Within a family the glyphs share a visual STYLE but are never identical:

    Synthetic / analytic  -- blue family,   round-ish markers  (o s ^ D v)
    HPO benchmarks        -- green family,  square-ish markers (s P X d p ...)
    Real-world (Olympus)  -- warm family,   pointed markers    (^ v < > * ...)

so you can tell the family at a glance from the hue, and the individual task
from the exact shape + shade.

Each task wedge is subdivided into THREE equal sub-sectors, one per LLM
backend: LLM | LLAMBO | LGBO.  Radial length = raw iteration count.
The GP arm is EXCLUDED -- not drawn, and counts are NOT renormalized.

Run from the project root (2_HybridBO/hybrid_bo):
    python scripts/generate_backend_circle.py
"""
import os, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ── NO TYPE 3 FONTS ─────────────────────────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["savefig.dpi"] = 600

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

PKG_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

# ── TASK LIST, GROUPED ──────────────────────────────────────────────────────
# Order here == order around the circle. Families stay contiguous.
TASKS_ANALYTIC = ["branin", "hartmann6", "rastrigin", "schwefel", "gallagher"]

TASKS_BAYESMARK = [f"bayesmark_{m}_{d}"
                   for m in ["SVM", "RF", "DT"]
                   for d in ["breast", "digits", "wine"]]
TASKS_YAHPO = [f"yahpo_lcbench_{i}" for i in ["3945", "7593", "34539"]]
TASKS_LASSO = ["lasso_diabetes", "lasso_breast_cancer"]
TASKS_HPO = TASKS_BAYESMARK + TASKS_YAHPO + TASKS_LASSO

TASKS_OLYMPUS = [
    "olympus_snar", "olympus_suzuki", "olympus_benzylation", "olympus_alkox",
    "olympus_colors_bob", "olympus_colors_n9", "olympus_fullerenes",
    "olympus_hplc", "olympus_photo_pce10", "olympus_photo_wf3",
]

# ── PER-FAMILY VISUAL STYLE ─────────────────────────────────────────────────
# Each family: a colour ramp (light -> dark within the family hue) and a pool
# of marker shapes with a shared "feel". Task i in the family gets
# (shape_pool[i % len], colour_ramp(i)).  Same style, never identical.
FAMILY_STYLE = {
    # blues, rounded / regular polygons
    "analytic": dict(
        ramp=["#9EC5E8", "#6BA3DA", "#4C72B0", "#31538A", "#1D3A66"],
        shapes=["o", "s", "^", "D", "v"],
    ),
    # greens/teals, boxy & plus-like
    "hpo": dict(
        ramp=["#BFE3C9", "#95D2A8", "#6EC08A", "#55A868", "#3F8A52",
              "#2F6E40", "#7FBFAE", "#4FA391", "#2E7F70", "#A7D8B4",
              "#79C295", "#4E9E6F", "#357F55", "#22633F"],
        shapes=["s", "P", "X", "d", "p", "h", "8", "*", "o", "^",
                "v", "<", ">", "D"],
    ),
    # warm reds/oranges/purples, pointed & starry
    "real": dict(
        ramp=["#F2C9A0", "#E8A96B", "#DD8452", "#C6653A", "#A94E2C",
              "#E8968F", "#D4635C", "#B4423E", "#C08AC0", "#96609B"],
        shapes=["^", "v", "<", ">", "*", "p", "h", "D", "X", "P"],
    ),
}

# family key -> (display name, task list)
GROUPS = [
    ("analytic", "Synthetic / analytic", TASKS_ANALYTIC),
    ("hpo",      "HPO benchmarks",       TASKS_HPO),
    ("real",     "Real-world (Olympus)", TASKS_OLYMPUS),
]

TASKS       = [t for _, _, lst in GROUPS for t in lst]
TASK_GROUP  = {t: g for g, _, lst in GROUPS for t in lst}
GROUP_TITLE = {g: n for g, n, _ in GROUPS}


def _ramp_color(ramp, i, n):
    """Pick colour i of n from a family ramp, interpolating if the ramp is
    shorter than the number of tasks so no two tasks collide."""
    if n <= len(ramp):
        return ramp[i]
    cmap = LinearSegmentedColormap.from_list("fam", ramp)
    return cmap(i / max(n - 1, 1))


# task -> (marker, color), built once
TASK_STYLE = {}
for _g, _name, _lst in GROUPS:
    _st = FAMILY_STYLE[_g]
    _n = len(_lst)
    for _i, _t in enumerate(_lst):
        TASK_STYLE[_t] = (
            _st["shapes"][_i % len(_st["shapes"])],
            _ramp_color(_st["ramp"], _i, _n),
        )

PRETTY = {
    "branin": "Branin", "hartmann6": "Hartmann6", "rastrigin": "Rastrigin",
    "schwefel": "Schwefel", "gallagher": "Gallagher",
    "lasso_diabetes": "Lasso diabetes",
    "lasso_breast_cancer": "Lasso breast",
}


def pretty(task):
    if task in PRETTY:
        return PRETTY[task]
    t = task
    t = t.replace("bayesmark_", "").replace("yahpo_lcbench_", "LCBench ")
    t = t.replace("olympus_", "").replace("photo_", "photo ")
    t = t.replace("colors_", "colors ").replace("_", " / ")
    return t


# ── CONFIG ──────────────────────────────────────────────────────────────────
LAMBDA      = 1.5
PICK_LAMBDA = "median"

SEEDS = list(range(10))

WINDOWS = [
    (1,  31, "a"),
    (31, 61, "b"),
]

BACKENDS       = ["llm", "llambo", "lgbo"]
BACKEND_LABELS = {"llm": "LLM", "llambo": "LLAMBO", "lgbo": "LGBO"}
ARMS_ALL       = ["gp", "llm", "llambo", "lgbo"]
PALETTE        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
ARM_COLOR      = {a: PALETTE[i] for i, a in enumerate(ARMS_ALL)}

# Geometry
GAP_FRAC     = 0.22
SUB_GAP_FRAC = 0.02
R0           = 0.18
BAR_EDGE_LW  = 0.0
SHOW_COUNTS  = False

# Rim annotation: one glyph per task, no text at all.
MARK_PAD   = 1.055      # glyph radius, in units of rtop -- keep tight
MARK_SIZE  = 8.5        # glyph size on the circle
MARK_EDGE  = 0.6        # thin dark outline keeps light shades legible
SHOW_IDX   = False      # set True to also print the task index next to it
IDX_PAD    = 1.135
FS_IDX     = 9.0

RMAX_FIXED = None
RGRID_STEP = None

# ── TYPOGRAPHY ──────────────────────────────────────────────────────────────
FS_RTICK     = 9
FS_LEGEND    = 15       # backend legend
FS_TASKLEG   = 11.5     # task legend
FS_TASKHEAD  = 12.5     # family headers in the task legend
FS_PANEL_TAG = 16
FS_CAPTION   = 10.5

LEG_MARK_SIZE = 10.0    # glyph size inside the task legend

# ── FIGURE PACKING ──────────────────────────────────────────────────────────
# Everything below is specified in INCHES and converted to figure fractions at
# the bottom. This is the whole trick: a polar plot draws a CIRCLE, so if the
# axes box is wider than it is tall the extra width is dead space that pushes
# the legend right and leaves a white strip at the page edge. By forcing the
# panel box to be SQUARE in inches, and then setting the figure width to
# exactly (left margin + circle + gap + legend + right margin), there is no
# strip left over to trim.
DPI = 600

FIG_H_IN     = 12.0     # total figure height
MARGIN_L_IN  = 0.85     # left margin -- this is what the (a)/(b) ylabel lives in
MARGIN_R_IN  = 0.55     # right margin after the legend
MARGIN_T_IN  = 0.75     # top strip for the backend legend
MARGIN_B_IN  = 0.35     # bottom strip for the caption
VGAP_IN      = 0.6    # vertical space between the two circles
HGAP_IN      = 0.15     # horizontal space between circle and task legend
LEGEND_W_IN  = 2.05     # width reserved for the task legend column

# One panel is square: its height sets its width, so the circle fills the box.
PANEL_H_IN = (FIG_H_IN - MARGIN_T_IN - MARGIN_B_IN - VGAP_IN) / 2
PANEL_W_IN = PANEL_H_IN

# Figure width falls out of the pieces -> no dead space by construction.
FIG_W_IN = (MARGIN_L_IN + PANEL_W_IN + HGAP_IN + LEGEND_W_IN + MARGIN_R_IN)
FIG_SIZE = (FIG_W_IN, FIG_H_IN)

# inches -> figure fractions
PANEL_L   = MARGIN_L_IN / FIG_W_IN
PANEL_W   = PANEL_W_IN / FIG_W_IN
PANEL_H   = PANEL_H_IN / FIG_H_IN
PANEL_GAP = VGAP_IN / FIG_H_IN
PANEL_TOP = 1.0 - MARGIN_T_IN / FIG_H_IN

# Task legend: single tall column, butted right against the circles.
TASKLEG_ANCHOR_X = (MARGIN_L_IN + PANEL_W_IN + HGAP_IN) / FIG_W_IN
TASKLEG_ANCHOR_Y = 0.52

# Backend legend + caption centred over the PANELS, not the whole figure.
PANEL_CX = PANEL_L + PANEL_W / 2

# (a)/(b) ylabel offset, in axes-fraction units. Must stay small enough that
# the text lands inside MARGIN_L_IN rather than off the canvas.
YLABEL_X = -0.10
# ────────────────────────────────────────────────────────────────────────────

MODEL      = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7

RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

_ONLINE_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)$"
)


# ════════════════════════════════════════════════════════════════════════════
# DISCOVERY + LOADERS
# ════════════════════════════════════════════════════════════════════════════
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


def per_iteration_arms(r):
    labels = []
    for j, arm_idx in enumerate(r.batch_arms):
        labels.extend([r.arm_names[arm_idx]] * int(r.batch_schedule[j]))
    return np.array(labels, dtype=object)


# ════════════════════════════════════════════════════════════════════════════
# COUNTING
# ════════════════════════════════════════════════════════════════════════════
def backend_counts(task, lam, iter_lo, iter_hi, cache):
    key = (task, lam)
    if key not in cache:
        on_disk = discover_seeds(task, lam)
        if not on_disk:
            cache[key] = None
        else:
            want = [s for s in SEEDS if s in on_disk]
            missing = [s for s in SEEDS if s not in on_disk]
            runs, kept = load_online(task, lam, want)
            cache[key] = None if not runs else (runs, kept, missing)

    if cache[key] is None:
        return None
    runs, kept, missing = cache[key]

    counts = {a: 0 for a in BACKENDS}
    gp_count = 0
    total = 0
    for r in runs:
        arms_t = per_iteration_arms(r)
        window = arms_t[iter_lo - 1: iter_hi]
        total += len(window)
        for a in window:
            if a in counts:
                counts[a] += 1
            elif a == "gp":
                gp_count += 1

    if total == 0:
        return None

    return counts, len(kept), gp_count, total, missing


def gather(iter_lo, iter_hi, cache):
    rows, skipped = [], []
    for task in TASKS:
        lam = LAMBDA if LAMBDA is not None else pick_lambda(task)
        if lam is None:
            skipped.append((task, "no lambda on disk"))
            continue
        res = backend_counts(task, lam, iter_lo, iter_hi, cache)
        if res is None:
            skipped.append((task, f"no runs at lambda={lam}"))
            continue
        counts, n, gp, total, missing = res
        marker, color = TASK_STYLE[task]
        rows.append(dict(task=task, title=pretty(task), lam=lam, counts=counts,
                         n=n, gp=gp, total=total, missing=missing,
                         group=TASK_GROUP[task], marker=marker, color=color))
    return rows, skipped


# ════════════════════════════════════════════════════════════════════════════
# DRAWING
# ════════════════════════════════════════════════════════════════════════════
def draw_panel(ax, rows, rmax, iter_lo, iter_hi, tag):
    n_tasks = len(rows)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    wedge   = 2 * np.pi / n_tasks
    gap     = wedge * GAP_FRAC
    usable  = wedge - gap
    sub_gap = wedge * SUB_GAP_FRAC
    sub_w   = (usable - 2 * sub_gap) / len(BACKENDS)

    r0   = R0 * rmax
    rtop = r0 + rmax

    for i, r in enumerate(rows):
        w_start = i * wedge + gap / 2
        for k, a in enumerate(BACKENDS):
            theta = w_start + k * (sub_w + sub_gap) + sub_w / 2
            val = r["counts"][a]
            ax.bar(theta, val, width=sub_w, bottom=r0,
                   color=ARM_COLOR[a], align="center", zorder=3,
                   edgecolor="white" if BAR_EDGE_LW else "none",
                   linewidth=BAR_EDGE_LW)
            if SHOW_COUNTS and val > 0:
                ax.text(theta, r0 + val + 0.02 * rmax, str(val),
                        ha="center", va="bottom", fontsize=7,
                        fontweight="bold", zorder=5)

        mid = i * wedge + wedge / 2

        # unique per-task glyph on the rim
        ax.plot([mid], [rtop * MARK_PAD],
                marker=r["marker"], markersize=MARK_SIZE,
                color=r["color"], markeredgecolor="0.15",
                markeredgewidth=MARK_EDGE,
                linestyle="none", clip_on=False, zorder=6)

        if SHOW_IDX:
            ax.text(mid, rtop * IDX_PAD, str(r["idx"]),
                    ha="center", va="center", fontsize=FS_IDX,
                    fontweight="bold", color="0.25", zorder=6)

    ax.set_ylim(0, rtop)
    step = RGRID_STEP if RGRID_STEP is not None else max(
        50, int(round(rmax / 4 / 50)) * 50)
    ticks = np.arange(0, rmax + 1, step)
    ax.set_yticks(r0 + ticks)
    ax.set_yticklabels([str(int(t)) for t in ticks],
                       fontsize=FS_RTICK, fontweight="bold", color="0.35")
    ax.set_rlabel_position(np.degrees(gap / 2) * 0.5)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.30, linewidth=0.6, zorder=0)
    ax.spines["polar"].set_visible(False)

    # panel identity as the y-label, pinned manually (polar axes place the
    # ylabel awkwardly by default)
    ax.set_ylabel(f"({tag})  $t$ = {iter_lo}–{iter_hi}",
                  fontsize=FS_PANEL_TAG, fontweight="bold", color="0.15",
                  rotation=90, labelpad=18, va="bottom")
    ax.yaxis.set_label_coords(YLABEL_X, 0.5)


def task_legend(fig, rows):
    """Per-task glyph -> name legend: ONE tall column on the right of the
    panels, families stacked with a blank spacer between them."""
    by_group = {g: [] for g, _, _ in GROUPS}
    for r in rows:
        by_group[r["group"]].append(r)

    blank = lambda: Line2D([], [], linestyle="none", marker="none")

    handles, labels, kinds = [], [], []
    for gi, (g, _, _) in enumerate(GROUPS):
        items = by_group[g]
        if not items:
            continue

        if gi > 0:                      # spacer between families
            handles.append(blank())
            labels.append(" ")
            kinds.append("pad")

        handles.append(blank())
        labels.append(GROUP_TITLE[g])
        kinds.append("head")

        for r in items:
            handles.append(Line2D([], [], linestyle="none",
                                  marker=r["marker"],
                                  markersize=LEG_MARK_SIZE,
                                  color=r["color"],
                                  markeredgecolor="0.15",
                                  markeredgewidth=MARK_EDGE))
            labels.append(r["title"])
            kinds.append("item")

    leg = fig.legend(handles, labels, loc="center left",
                     bbox_to_anchor=(TASKLEG_ANCHOR_X, TASKLEG_ANCHOR_Y),
                     ncol=1, frameon=False,
                     fontsize=FS_TASKLEG, handlelength=1.3,
                     handletextpad=0.6, labelspacing=0.32,
                     borderpad=0.0)

    for txt, kind in zip(leg.get_texts(), kinds):
        txt.set_fontweight("bold")
        if kind == "head":
            txt.set_fontsize(FS_TASKHEAD)
            txt.set_color("0.05")
        else:
            txt.set_color("0.25")
    return leg


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    cache = {}
    panels = []
    for lo, hi, tag in WINDOWS:
        rows, skipped = gather(lo, hi, cache)
        if not rows:
            print(f"[skip] window t={lo}-{hi}: no task had results")
            continue
        panels.append(dict(rows=rows, lo=lo, hi=hi, tag=tag, skipped=skipped))

    if not panels:
        print("[abort] nothing to plot")
        return

    ref_order = [r["task"] for r in panels[0]["rows"]]
    idx_of = {t: i + 1 for i, t in enumerate(ref_order)}
    for p in panels:
        for r in p["rows"]:
            r["idx"] = idx_of.get(r["task"], 0)

    for p in panels:
        print(f"=== t={p['lo']}-{p['hi']}: {len(p['rows'])} tasks plotted, "
              f"{len(p['skipped'])} skipped ===")
        for r in p["rows"]:
            warn = f"  [missing seeds {r['missing']}]" if r["missing"] else ""
            shown = sum(r["counts"].values())
            bars = "  ".join(f"{BACKEND_LABELS[a]}={r['counts'][a]:>4}"
                             for a in BACKENDS)
            print(f"{r['idx']:>2}. {r['title']:<18} lam={r['lam']:<5} "
                  f"n={r['n']:<3} {bars}   [gp {r['gp']}]  "
                  f"{shown}/{r['total']}{warn}")
        print()

    data_rmax = max(max(max(r["counts"][a] for a in BACKENDS)
                        for r in p["rows"]) for p in panels)
    rmax = RMAX_FIXED if RMAX_FIXED is not None else data_rmax
    if RMAX_FIXED is not None and data_rmax > RMAX_FIXED:
        print(f"  [warn] a bar ({data_rmax}) exceeds RMAX_FIXED={RMAX_FIXED}; "
              f"it will be clipped at the outer ring.")

    # ── explicit panel rectangles: guaranteed no overlap ────────────────────
    fig = plt.figure(figsize=FIG_SIZE)
    axes = []
    for i in range(len(panels)):
        top = PANEL_TOP - i * (PANEL_H + PANEL_GAP)
        axes.append(fig.add_axes([PANEL_L, top - PANEL_H, PANEL_W, PANEL_H],
                                 projection="polar"))

    for ax, p in zip(axes, panels):
        draw_panel(ax, p["rows"], rmax, p["lo"], p["hi"], p["tag"])

    backend_leg = fig.legend(
        handles=[Patch(facecolor=ARM_COLOR[a], label=BACKEND_LABELS[a])
                 for a in BACKENDS],
        ncol=len(BACKENDS), loc="upper center",
        bbox_to_anchor=(PANEL_CX, 0.999),
        frameon=False, handlelength=1.6,
        columnspacing=2.0, handletextpad=0.6,
        prop={"size": FS_LEGEND, "weight": "bold"})
    fig.add_artist(backend_leg)

    task_legend(fig, panels[0]["rows"])

    # fig.text(PANEL_CX, 0.012,
    #          f"radius = iterations chosen (seeds {SEEDS[0]}–{SEEDS[-1]}); "
    #          f"GP excluded",
    #          ha="center", fontsize=FS_CAPTION, fontweight="bold", color="0.35")

    outdir = FIG_ROOT / "main"
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"backend_circle.{ext}"
        # NOTE: no bbox_inches="tight" -- it would re-crop and undo the
        # explicit panel/legend placement above.
        fig.savefig(path, dpi=DPI)
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()