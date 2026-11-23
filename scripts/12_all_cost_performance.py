#!/usr/bin/env python3
"""
Fig 8 -- 2x4 AAAI-style ZOOMED grid, one hand-picked (task, lambda, seed) per
panel, ONE shared legend for the whole figure.

Follows the single-panel zoomed script exactly: every panel is restricted to
[0, ZOOM_XMAX] (default 0-0.03) unless overridden per-panel. Curves that run
past that limit (notably LLAMBO) are simply CLIPPED by the x-limit -- the
underlying data is NOT truncated or resampled, so the visible portion is
exactly what a full-range plot would show over the same interval. Each panel
annotates the methods whose trajectories are incomplete inside the window,
reporting where they actually end.

Layout: PANELS is read in order, filled row-major into a 2x4 grid (8 panels).
Subplot titles are the TASK NAME ONLY (no lambda, no seed, no "Fig 8").

One legend is built from the UNION of series across all panels (so a method
missing from one panel doesn't break the shared legend) and drawn once above
the whole grid, in a fixed order: Online first, then baselines in BASELINES
order.

y-axis: log10 simple regret, NORMAL orientation (down = better).
Step plots, same f_best_curve / cost_curve as the original.

Drop next to generate_figures.py (same scripts/ folder) and run:
    python scripts/generate_fig8_grid.py

Outputs:
    results/figures/main/fig8_grid.pdf
    results/figures/main/fig8_grid.png
"""
import os
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
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["savefig.dpi"] = 600
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# -- Same project root logic as generate_figures.py --------------------------
PKG_ROOT = Path(__file__).resolve().parent.parent      # .../2_HybridBO/hybrid_bo
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

RESULTS_DIR = PKG_ROOT / "results"
FIG_ROOT    = RESULTS_DIR / "figures"
store = ResultStore(root_dir=str(RESULTS_DIR))

# -- Config ------------------------------------------------------------------
MODEL      = "deepseek-v4-flash"
MAIN_ALPHA = 0.01
MAIN_ETA   = 7
BASELINES  = ["gp", "llm", "llambo", "lgbo", "smac", "tpe"]

# ---------------------------------------------------------------------------
# THE EIGHT PANELS. Order = reading order, filled row-major into a 2x4 grid.
#   zoom : optional per-panel x-limit override; None -> use ZOOM_XMAX
# ---------------------------------------------------------------------------
PANELS = [
    dict(task="hartmann6",             lam=10.0, seed=6, title="Hartmann6",         zoom=None),
    dict(task="schwefel",              lam=10.0,  seed=2, title="Schwefel",          zoom=None),
    dict(task="yahpo_lcbench_7593",    lam=10,   seed=6, title="LCBench-7593",      zoom=None),
    dict(task="olympus_suzuki",        lam=10.0, seed=3, title="Olympus Suzuki",    zoom=None),
    dict(task="bayesmark_SVM_digits",  lam=10.0, seed=8, title="SVM Digits",        zoom=None),
    dict(task="bayesmark_SVM_breast",  lam=10.0, seed=7, title="SVM Breast",        zoom=None),
    dict(task="lasso_diabetes",        lam=10.0, seed=6, title="Lasso Diabetes",    zoom=None),
    dict(task="lasso_breast_cancer",   lam=10.0, seed=2, title="Lasso Breast Cancer", zoom=None),
]

GRID_NROWS = 2
GRID_NCOLS = 4

# Shared zoom window for every panel (same role as ZOOM_XMAX in the 1-panel script).
# Set to None to auto-fit each panel to ALL its data (incl. LLAMBO's tail).
ZOOM_XMAX = 0.03

# Annotate curves that extend beyond the zoom window.
ANNOTATE_CLIPPED = True
OVERRUN_TOL = 1e-9          # ignore float-noise overshoot

# y-limits from the VISIBLE window only, so a panel doesn't waste vertical
# space on regret values that occur off-panel.
YLIM_FROM_VISIBLE = True

# Only label outer axes (cleaner grid). Set False to label every panel.
SHARE_AXIS_LABELS = True

FSTAR_SCAN_SEEDS = list(range(0, 10))
REGRET_FLOOR = 1e-6         # log10 can't take 0
OUT_STEM = "fig8_grid"

# -- Style: AAAI-ish, colorblind-friendly (Okabe-Ito) -------------------------
ONLINE_STYLE = dict(color="black", lw=2.2, ls="-", zorder=5,
                    marker="o", markersize=3.8, markevery=0.14,
                    markerfacecolor="black", markeredgecolor="black")

BASE_STYLE = {
    "gp":     dict(color="#0072B2", lw=1.1, ls="--",  marker="s", markersize=2.6),
    "llm":    dict(color="#E69F00", lw=1.1, ls="-.",  marker="^", markersize=2.6),
    "llambo": dict(color="#009E73", lw=1.1, ls=":",   marker="D", markersize=2.4),
    "lgbo":   dict(color="#CC79A7", lw=1.1, ls=(0, (3, 1, 1, 1)),
                   marker="v", markersize=2.6),
    "smac":   dict(color="#D55E00", lw=1.1, ls=(0, (5, 1)),
                   marker="P", markersize=2.6),
    "tpe":    dict(color="#56B4E9", lw=1.1, ls=(0, (1, 1)),
                   marker="X", markersize=2.6),
}
BASE_LABEL = {"gp": "GP", "llm": "LLM", "llambo": "LLAMBO", "lgbo": "LGBO", "tpe": "TPE", "smac": "SMAC"}
ONLINE_LABEL = "HybridBO"

# Note text colored to match its curve, so you can tell which line got cut.
NOTE_COLOR = {BASE_LABEL[k]: v["color"] for k, v in BASE_STYLE.items()}
NOTE_COLOR[ONLINE_LABEL] = "black"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def online_opt_name(lam, alpha=MAIN_ALPHA, eta=MAIN_ETA):
    return f"online_hybrid_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}"


def load_one(task, opt, seed):
    try:
        if store.exists(task, opt, MODEL, seed):
            return store.load(task, opt, MODEL, seed)
    except Exception:
        pass
    return None


# -- f* reference, per task --------------------------------------------------
_FSTAR_CACHE = {}


def task_fstar(task):
    try:
        from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
        t = OnlineExperimentRunner._make_task(task)
        fs = t.f_star()
        return None if fs is None else float(fs)
    except Exception:
        return None


def fstar_for(task, lam):
    """True f* if the task exposes one; else best f_best seen across this task's runs."""
    if task in _FSTAR_CACHE:
        return _FSTAR_CACHE[task]

    fs = task_fstar(task)
    if fs is None:
        best = -np.inf
        name = online_opt_name(lam)
        for s in FSTAR_SCAN_SEEDS:
            r = load_one(task, name, s)
            if r is not None:
                best = max(best, float(np.max(r.f_best_curve)))
        for opt in BASELINES:
            for s in FSTAR_SCAN_SEEDS:
                r = load_one(task, opt, s)
                if r is not None:
                    best = max(best, float(np.max(r.f_best_curve)))
        fs = best if np.isfinite(best) else None

    _FSTAR_CACHE[task] = fs
    return fs


def curve_xy(r, fstar):
    """(cumulative cost, log10 simple regret) for one run. Data untouched."""
    f = np.asarray(r.f_best_curve, dtype=float)
    c = np.asarray(r.cost_curve, dtype=float)
    L = min(len(f), len(c))
    reg = np.clip(fstar - f[:L], REGRET_FLOOR, None)
    return c[:L], np.log10(reg)


def build_series(task, lam, seed, fstar):
    """List of (label, x, y, style). Online first, then baselines."""
    series = []
    on = load_one(task, online_opt_name(lam), seed)
    if on is not None:
        x, y = curve_xy(on, fstar)
        series.append((ONLINE_LABEL, x, y, dict(ONLINE_STYLE)))
    for opt in BASELINES:
        r = load_one(task, opt, seed)
        if r is None:
            continue
        x, y = curve_xy(r, fstar)
        kw = dict(BASE_STYLE[opt])
        kw.update(markevery=0.2, alpha=0.95, zorder=3)
        series.append((BASE_LABEL[opt], x, y, kw))
    return series


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", lw=0.4, color="0.85", alpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def annotate_overruns(ax, overruns):
    """overruns: list of (label, final_cost) for methods running past the window.

    Stacked bottom-right, one short line each, colored to match the curve.
    """
    if not overruns:
        return
    overruns = sorted(overruns, key=lambda t: -t[1])   # most expensive first
    for i, (label, cost) in enumerate(overruns):
        ax.annotate(
            f"{label} extends to \\${cost:.3f}",
            xy=(0.975, 0.035 + 0.075 * i),
            xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=5.6,
            color=NOTE_COLOR.get(label, "0.35"),
            alpha=0.9,
        )


def draw_panel(ax, spec):
    """Returns (handles, labels, overruns) for THIS panel."""
    task, lam, seed = spec["task"], spec["lam"], spec["seed"]
    fstar = fstar_for(task, lam)

    zoom = spec.get("zoom") or ZOOM_XMAX

    if fstar is None:
        ax.text(0.5, 0.5, f"no f* for\n{task}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.5")
        _style_axis(ax)
        if zoom is not None:
            ax.set_xlim(0, zoom)
        ax.set_title(spec["title"], pad=4)
        return [], [], []

    series = build_series(task, lam, seed, fstar)
    if not series:
        ax.text(0.5, 0.5, f"no data\n{task} (λ={lam:g}, seed={seed})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color="0.5")
        _style_axis(ax)
        if zoom is not None:
            ax.set_xlim(0, zoom)
        ax.set_title(spec["title"], pad=4)
        return [], [], []

    for label, x, y, kw in series:
        ax.step(x, y, where="post", label=label, **kw)

    _style_axis(ax)

    # Zoom window; None -> auto-fit this panel's full data.
    xmax = zoom
    if xmax is None:
        xmax = max(float(np.max(s[1])) for s in series) * 1.03
    ax.set_xlim(0, xmax)        # curves past xmax are CLIPPED, not truncated

    # y-limits from points VISIBLE inside the zoom window.
    if YLIM_FROM_VISIBLE:
        vis = []
        for _, x, y, _ in series:
            m = x <= xmax
            if m.any():
                vis.append(y[m])
        all_y = np.concatenate(vis) if vis else np.concatenate([s[2] for s in series])
    else:
        all_y = np.concatenate([s[2] for s in series])
    ylo, yhi = float(np.min(all_y)), float(np.max(all_y))
    pad = 0.06 * (yhi - ylo) if yhi > ylo else 0.5
    ax.set_ylim(ylo - pad, yhi + pad)

    ax.set_title(rf"{spec['title']} $\lambda={spec['lam']:g}$ (seed={spec['seed']})", pad=4)

    # Every method whose trajectory is incomplete within the zoom window.
    overruns = [(lbl, float(np.max(x)))
                for lbl, x, _, _ in series
                if float(np.max(x)) > xmax + OVERRUN_TOL]

    h, l = ax.get_legend_handles_labels()
    return h, l, overruns


def main():
    print(f"Project root: {PKG_ROOT}")
    zoom_desc = f"[0, {ZOOM_XMAX:g}]" if ZOOM_XMAX is not None else "auto"
    print(f"Zoom window: {zoom_desc}\n")

    n_panels = len(PANELS)
    n_slots = GRID_NROWS * GRID_NCOLS
    if n_panels > n_slots:
        print(f"[warn] {n_panels} panels but grid only has {n_slots} slots "
              f"({GRID_NROWS}x{GRID_NCOLS}); extra panels will be dropped.")

    fig, axes = plt.subplots(GRID_NROWS, GRID_NCOLS,
                              figsize=(4.0 * GRID_NCOLS, 3.0 * GRID_NROWS),
                              squeeze=False)
    flat_axes = axes.flatten()

    # Collect handles/labels across ALL panels so the shared legend is
    # complete even if a given panel is missing a method.
    handle_by_label = {}

    for idx, ax in enumerate(flat_axes):
        if idx >= n_panels:
            ax.axis("off")   # unused slot
            continue

        spec = PANELS[idx]
        print(f"  panel[{idx}]: {spec['task']}  lambda={spec['lam']:g}  seed={spec['seed']}")
        h, l, overruns = draw_panel(ax, spec)

        for hh, ll in zip(h, l):
            if ll not in handle_by_label:
                handle_by_label[ll] = hh

        if ANNOTATE_CLIPPED:
            annotate_overruns(ax, overruns)
        for lbl, cost in sorted(overruns, key=lambda t: -t[1]):
            print(f"    [clipped] {lbl} extends to ${cost:.4f}")

        # Axis labels: only outer edges if SHARE_AXIS_LABELS, else every panel.
        row, col = divmod(idx, GRID_NCOLS)
        is_bottom = (row == GRID_NROWS - 1) or (idx + GRID_NCOLS >= n_panels)
        is_left = (col == 0)
        if SHARE_AXIS_LABELS:
            if is_bottom:
                ax.set_xlabel("Cumulative LLM cost")
            if is_left:
                ax.set_ylabel(r"$\log_{10}$ regret")
        else:
            ax.set_xlabel("Cumulative LLM cost")
            ax.set_ylabel(r"$\log_{10}$ regret")

    # -- ONE shared legend for the whole grid, fixed order: Online, then baselines --
    ordered_labels = [ONLINE_LABEL] + [BASE_LABEL[o] for o in BASELINES]
    handles, labels = [], []
    for lbl in ordered_labels:
        if lbl in handle_by_label:
            handles.append(handle_by_label[lbl])
            labels.append(lbl)

    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), ncol=len(labels),
                   frameon=False, handlelength=2.2, handletextpad=0.55,
                   columnspacing=1.3)

    fig.tight_layout()

    outdir = FIG_ROOT / "main"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / OUT_STEM
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=1000, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  saved {stem}.pdf / .png")


if __name__ == "__main__":
    main()