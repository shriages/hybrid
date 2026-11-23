#!/usr/bin/env python3
r"""
Ablation table with f* SHARED with the main tables (pooled over both backends).

    OUTPUT  results/ablation/tables/ablation_all_<ts>.csv
            results/ablation/tables/ablation_all_<ts>.tex   (paste-ready)

WHAT THIS FIXES
---------------
The original generate_ablation_table.py computes f* from the ablation runs
alone (pool-relative within the ablation). That puts ablation regret on a
DIFFERENT scale than the main cost-performance table, and makes the
"Full HybridBO" row here disagree with the same method's regret there.

This version anchors regret on the SAME f* the main tables use: the exact
optimum where the task provides one, else the best f_best pooled over BOTH
backends (FSTAR_MODELS) across the main sweep -- and, optionally, the
ablation variants too (POOL_ABLATION_INTO_FSTAR). The anchor is identical to
21_cost_performance_tpe.py, so Full's regret matches Table 1 cell-for-cell.

HOW THE RUNS ARE READ  (learned from generate_ablation_table.py)
----------------------------------------------------------------
  * Full HybridBO   -> MAIN sweep, results/raw/, at LAMBDA_FULL[task],
                       dir  online_hybrid_lambda_L_alpha_A_eta_E   (no suffix)
  * Ablation variants -> results/ablation/raw/, at the SAME lambda, each
                       differing from Full by one override. The directory name
                       is rebuilt by variant_storage_name(), copied verbatim
                       from the original script so names match on disk:
        no_exploration  -> alpha=0        ..._alpha_0.00_...
        fixed_distribution -> eta=0       ..._eta_0.00
        no_gp/llm/llambo/lgbo -> arms=... ..._arms_a-b-c
        one_time_*      -> switch suffix  ..._<switch_mode>
        no_cost         -> lambda=0       (its OWN lambda, not LAMBDA_FULL)
  Two ResultStore roots, one per tree, exactly as the original.

f* is pooled over the MAIN sweep via a small, dependency-free reimplementation
of 21's fstar_for(): discover the main online dirs + baselines, take the max
f_best over both backends. If the main table already emitted an fstar_<ts>.csv
you can point FSTAR_CSV at it and skip recomputation entirely (preferred:
guarantees byte-identical anchors).

Usage
-----
    python scripts/27_ablation_shared_fstar.py
    python scripts/27_ablation_shared_fstar.py --tasks hartmann6 bayesmark_SVM_breast olympus_fullerenes
    python scripts/27_ablation_shared_fstar.py --fstar-csv results/logs/fstar_20260723_144409.csv
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
import re
from pathlib import Path
from datetime import datetime

import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

MAIN_ROOT = PKG_ROOT / "results"
ABL_ROOT  = PKG_ROOT / "results" / "ablation"
OUT_TAB   = ABL_ROOT / "tables"
OUT_LOG   = ABL_ROOT / "logs"

# ── Shared-anchor config (mirror 21_cost_performance_tpe.py) ────────────────
MODEL_TABULATED = "deepseek-v4-flash"     # which backend's runs fill the table
FSTAR_MODELS = ["deepseek-v4-flash", "gpt-5.4-mini"]   # f* pools over both

# ── Reference f* (HARDCODED, from the OLD main-table export) ─────────────────
# These are the EXACT f* values from results/logs/fstar_20260724_142616.csv
# (the OpenAI/DeepSeek pooled run you used for the published main table). They
# are hardcoded so regret here is anchored to the SAME reference you used
# before -- you can eyeball this dict against that CSV to confirm.
#
# NOTE this is NOT recomputed from the ablation runs. A variant that beats
# this f* simply gets NEGATIVE regret, which is expected and fine.
#
# source column in the CSV, for the record:
#   hartmann6, schwefel            -> exact
#   everything else                -> pooled(2 backends)
FSTAR_REFERENCE = {
    "lasso_breast_cancer":  -0.260974709215,
    "yahpo_lcbench_7593":   87.2660675049,
    "hartmann6":            3.32237,
    "schwefel":             0.0,
    "bayesmark_SVM_digits": 0.993041037553,
    "bayesmark_SVM_breast": 0.949450549451,
    "lasso_diabetes":       -0.648716027044,
    "olympus_suzuki":       101.391703555,
}

# If a task is NOT in FSTAR_REFERENCE, fall back to recomputing from the main
# sweep (pooled over FSTAR_MODELS). Set STRICT_REFERENCE=True to instead error
# out, guaranteeing every regret uses a value you have verified by hand.
STRICT_REFERENCE = True

# If set, load f* from this CSV instead of FSTAR_REFERENCE above. Same columns
# as the export (task, source, fstar). Leave None to use the hardcoded dict.
FSTAR_CSV = None      # e.g. "results/logs/fstar_20260724_142616.csv"

# Fold the ablation variants' own f_best into the max, too? The main tables do
# NOT see the ablation runs, so leaving this False keeps the anchor identical
# to Table 1. Set True only if a variant might exceed every main run and you
# want regret measured against the true best-seen.
POOL_ABLATION_INTO_FSTAR = False

# ── Ablation base config (must match modal_ablation.py header) ──────────────
ALPHA = 0.01
ETA   = 7
FULL_POOL = ["gp", "llm", "llambo", "lgbo"]
LAM = 10.0

# Seed policy. The main cost-performance table averages Full over the seeds
# that exist for the Full run (n=10 there), NOT a fixed [0..4]. Hardcoding 5
# seeds here made Full's regret/cost disagree with Table 1. Options:
#   "match_full"  : use exactly the seeds present for the Full run on disk,
#                   so Full's numbers reproduce Table 1. Each variant is then
#                   averaged over the seeds IT has that intersect Full's.
#   "fixed"       : use SEEDS_FIXED below (the old behaviour).
# Variants may have been run for fewer seeds than Full; SEED_POLICY controls
# only how Full is anchored, and each variant reports its own available seeds.
SEED_POLICY = "match_full"
SEEDS_FIXED = [0, 1, 2, 3, 4]
SEED_SCAN = list(range(20))          # seeds to probe on disk when matching

# Main-sweep baselines + online pools that define f* (mirror script 21).
MAIN_BASELINES = ["gp", "tpe", "smac", "llm", "lgbo", "llambo"]

# Tasks for THIS table and the lambda that defines Full for each.
TASKS = ["bayesmark_SVM_breast",
         "bayesmark_SVM_digits", "lasso_breast_cancer",
         ]
LAMBDA_FULL = {t: LAM for t in TASKS}

# Rows to show, in order. Comment out any you don't want (the target table
# omits "No cost term" and the three one-time switches).
VARIANTS = [
    # (row label,                       variant key,        note)
    # ("No cost term",                  "no_cost",          "lambda=0"),
    ("No exploration",                  "no_exploration",   "alpha=0"),
    ("Fixed distribution",              "fixed_distribution", "eta=0"),
    ("__RULE__",                        None,               None),
    ("Remove GP arm",                   "no_gp",            ""),
    ("Remove LLM arm",                  "no_llm",           ""),
    ("Remove LLAMBO arm",               "no_llambo",        ""),
    ("Remove LGBO arm",                 "no_lgbo",          ""),
]
LAMBDA_FIXED_VARIANTS = {"no_cost"}

PRETTY_TASK = {
    "hartmann6": "hartmann6",
    "bayesmark_SVM_breast": "SVM breast",
    "olympus_fullerenes": "olympus\\_fullerene",
}


# ── Storage-name reconstruction (verbatim from the original) ────────────────
def _variant_overrides(variant: str) -> dict:
    return {
        "no_cost":            dict(lambda_=0.0),
        "no_exploration":     dict(alpha=0.0),
        "fixed_distribution": dict(eta=0.0),
        "one_time_llm":       dict(switch_mode="one_time_llm"),
        "one_time_llambo":    dict(switch_mode="one_time_llambo"),
        "one_time_lgbo":      dict(switch_mode="one_time_lgbo"),
        "no_gp":              dict(arms=["llm", "llambo", "lgbo"]),
        "no_llm":             dict(arms=["gp", "llambo", "lgbo"]),
        "no_llambo":          dict(arms=["gp", "llm", "lgbo"]),
        "no_lgbo":            dict(arms=["gp", "llm", "llambo"]),
    }[variant]


def full_storage_name(lam: float) -> str:
    return f"online_hybrid_lambda_{lam:.2f}_alpha_{ALPHA:.2f}_eta_{ETA:.2f}"


def variant_storage_name(variant: str, lam: float) -> str:
    o = _variant_overrides(variant)
    eff_lam   = o.get("lambda_", lam)
    eff_alpha = o.get("alpha", ALPHA)
    eff_eta   = o.get("eta", ETA)
    base = (f"online_hybrid_lambda_{eff_lam:.2f}"
            f"_alpha_{eff_alpha:.2f}_eta_{eff_eta:.2f}")
    if "switch_mode" in o:
        return f"{base}_{o['switch_mode']}"
    arms = o.get("arms", FULL_POOL)
    if list(arms) != FULL_POOL:
        base += "_arms_" + "-".join(arms)
    return base


# ── Loading ─────────────────────────────────────────────────────────────────
_main_store = ResultStore(root_dir=str(MAIN_ROOT))
_abl_store  = ResultStore(root_dir=str(ABL_ROOT))


def _load(store, task, opt_name, model, seeds):
    out = []
    for s in seeds:
        try:
            if store.exists(task, opt_name, model, s):
                out.append(store.load(task, opt_name, model, s))
        except Exception:
            pass
    return out


def _seeds_present(store, task, opt_name, model, scan=SEED_SCAN):
    """Seeds that actually exist on disk for (task, opt, model)."""
    return [s for s in scan if store.exists(task, opt_name, model, s)]


def full_seeds(task, lam):
    """Seed set that defines Full, matching how the main table averaged it.

    match_full -> exactly the seeds on disk for the Full run, so Full's
    regret/cost reproduce Table 1. fixed -> the old blanket list.
    """
    if SEED_POLICY == "fixed":
        return list(SEEDS_FIXED)
    return _seeds_present(_main_store, task, full_storage_name(lam),
                          MODEL_TABULATED)


# ── Shared f* (mirror of 21_cost_performance_tpe.py) ────────────────────────
_FSTAR_CACHE = {}
_FSTAR_FROM_CSV = None


def _load_fstar_csv(path):
    global _FSTAR_FROM_CSV
    _FSTAR_FROM_CSV = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = r.get("fstar", "")
            _FSTAR_FROM_CSV[r["task"]] = float(v) if v not in ("", None) else None
    print(f"f* read from {path} ({len(_FSTAR_FROM_CSV)} tasks)")


def _exact_fstar(task):
    try:
        from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
        t = OnlineExperimentRunner._make_task(task)
        fs = t.f_star()
        return None if fs is None else float(fs)
    except Exception:
        return None


_ONLINE_RE = re.compile(
    r"^online_hybrid_lambda_[\d.]+_alpha_[\d.]+_eta_[\d.]+")


def _main_online_dirs(task):
    d = MAIN_ROOT / "raw" / task
    if not d.exists():
        return []
    return [p.name for p in d.iterdir()
            if p.is_dir() and _ONLINE_RE.match(p.name)]


def shared_fstar(task, ablation_pool=None):
    """f* for regret, in priority order:
      1. FSTAR_CSV, if provided (loaded into _FSTAR_FROM_CSV)
      2. the hardcoded FSTAR_REFERENCE dict (the value you used before)
      3. recompute from the main sweep, pooled over both backends
         -- ONLY if STRICT_REFERENCE is False.

    This is deliberately anchored to your OLD table's f*, NOT recomputed from
    the ablation runs. A variant beating this f* yields negative regret, which
    is expected.
    """
    if _FSTAR_FROM_CSV is not None and task in _FSTAR_FROM_CSV:
        return _FSTAR_FROM_CSV[task]
    if task in FSTAR_REFERENCE:
        return FSTAR_REFERENCE[task]

    if STRICT_REFERENCE:
        sys.exit(
            f"[fstar] no reference value for '{task}'. Add it to "
            f"FSTAR_REFERENCE (copy from your fstar_*.csv), or pass "
            f"--fstar-csv, or set STRICT_REFERENCE=False to recompute.")

    if task in _FSTAR_CACHE:
        return _FSTAR_CACHE[task]
    fs = _exact_fstar(task)
    if fs is None:
        best = -np.inf
        for model in FSTAR_MODELS:
            for dirname in _main_online_dirs(task):
                for r in _load(_main_store, task, dirname, model, SEED_SCAN):
                    best = max(best, float(r.f_best))
            for opt in MAIN_BASELINES:
                for r in _load(_main_store, task, opt, model, SEED_SCAN):
                    best = max(best, float(r.f_best))
        if POOL_ABLATION_INTO_FSTAR and ablation_pool:
            for runs in ablation_pool:
                for r in runs:
                    best = max(best, float(r.f_best))
        fs = best if np.isfinite(best) else None
    _FSTAR_CACHE[task] = fs
    return fs


def regret_of(runs, fstar):
    if fstar is None:
        return np.array([np.nan] * len(runs))
    return np.array([fstar - float(r.f_best) for r in runs])


def cost_of(runs):
    return np.array([float(r.total_cost) for r in runs])


# ── Build one task's column ─────────────────────────────────────────────────
def build_task(task, log):
    lam = LAMBDA_FULL.get(task, LAM)

    fseeds = full_seeds(task, lam)
    full_runs = _load(_main_store, task, full_storage_name(lam),
                      MODEL_TABULATED, fseeds)
    if not full_runs:
        log.write(f"[skip] {task}: Full not found at lambda={lam} "
                  f"(seeds tried: {fseeds})\n")
        print(f"[skip] {task}: Full run not found at lambda={lam}")
        return None
    log.write(f"  {task}: Full seeds = {fseeds} (n={len(full_runs)})\n")

    loaded = {}
    for label, key, note in VARIANTS:
        if key is None:
            continue
        vlam = 0.0 if key in LAMBDA_FIXED_VARIANTS else lam
        vname = variant_storage_name(key, vlam)
        # Each variant is averaged over the seeds IT has on disk. Variants may
        # have fewer seeds than Full; that is reported per row via n.
        vseeds = (_seeds_present(_abl_store, task, vname, MODEL_TABULATED)
                  if SEED_POLICY == "match_full" else SEEDS_FIXED)
        runs = _load(_abl_store, task, vname, MODEL_TABULATED, vseeds)
        loaded[key] = (vname, runs)
        if not runs:
            log.write(f"  [missing] {task}/{label}: no runs at {vname} "
                      f"(seeds tried: {vseeds})\n")
        else:
            log.write(f"  {task}/{label}: seeds={vseeds} n={len(runs)}\n")

    ablation_pool = [r for _, r in loaded.values() if r]
    fstar = shared_fstar(task, ablation_pool)

    def stats(runs):
        if not runs:
            return dict(n=0, regret_mean=np.nan, regret_std=np.nan,
                        cost_mean=np.nan, cost_std=np.nan)
        R, C = regret_of(runs, fstar), cost_of(runs)
        return dict(n=len(runs),
                    regret_mean=float(np.nanmean(R)),
                    regret_std=float(np.nanstd(R)),
                    cost_mean=float(np.mean(C)),
                    cost_std=float(np.std(C)))

    col = {"__full__": dict(label="Full HybridBO", **stats(full_runs))}
    for label, key, note in VARIANTS:
        if key is None:
            continue
        col[key] = dict(label=label, **stats(loaded[key][1]))
    return dict(task=task, lam=lam, fstar=fstar, col=col)


# ── LaTeX (matches the target layout: tasks as column pairs) ────────────────
def fmt(x, dec):
    """Numeric value formatted to `dec` decimals, no math wrapper. Missing
    stays a bare '--'."""
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:.{dec}f}"


# Row keys forming the "Remove arm" group ranked in rule 2.
REMOVE_KEYS = ["no_gp", "no_llm", "no_llambo", "no_lgbo"]

# Direction arrows are purely descriptive: they say whether a variant moved
# the metric UP or DOWN relative to Full, not whether that change is good.
ARROW_UP = r"\uparrow"
ARROW_DOWN = r"\downarrow"
EPS = 1e-12


def cell(value, dec, full_value=None, rank=None):
    r"""One table cell.

    arrow -- if full_value is given and value is finite: append \uparrow when
             value > Full, \downarrow when value < Full (same task/column).
             Full's own row passes full_value=None, so it gets no arrow.
    rank  -- among the Remove-arm rows only: 1 => \mathbf (largest),
             2 => \underline (second largest). Wrapped in $...$ so arrow and
             bold/underline compose.
    """
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    body = f"{value:.{dec}f}"
    if full_value is not None and not (isinstance(full_value, float)
                                       and full_value != full_value):
        if value > full_value + EPS:
            body += f"\\,{ARROW_UP}"
        elif value < full_value - EPS:
            body += f"\\,{ARROW_DOWN}"
    if rank == 1:
        return rf"$\mathbf{{{body}}}$"
    # if rank == 2:
    #     return rf"$\underline{{{body}}}$"
    return f"${body}$"


def _rank_largest(values):
    """1 = largest, 2 = second largest; ties share the higher rank. NaNs
    excluded. Returns {index: rank} with only ranks 1 and 2 populated."""
    idx = [i for i, v in enumerate(values)
           if v is not None and not (isinstance(v, float) and v != v)]
    order = sorted(idx, key=lambda i: values[i], reverse=True)
    out, rank, prev = {}, 0, None
    for i in order:
        if prev is None or abs(values[i] - prev) > EPS:
            rank += 1
            prev = values[i]
        if rank <= 2:
            out[i] = rank
    return out


def render(tabs, caption, label, regret_dec=4, cost_dec=5):
    tasks = [t["task"] for t in tabs]
    ntask = len(tasks)

    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         rf"\begin{{tabular}}{{l {' '.join(['cc'] * ntask)}}}",
         r"\toprule"]
    L.append("".join(rf"& \multicolumn{{2}}{{c}}{{{PRETTY_TASK.get(t, t)}}} "
                     for t in tasks).rstrip() + r" \\")
    L.append("".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                     for i in range(ntask)))
    L.append("Variant " + r"& Regret & Cost " * ntask + r"\\")
    L.append(r"\midrule")

    # Full's per-task values -- the reference the arrows point against.
    full_reg = {t["task"]: t["col"]["__full__"]["regret_mean"] for t in tabs}
    full_cost = {t["task"]: t["col"]["__full__"]["cost_mean"] for t in tabs}

    # Rule-2 ranks: per task, per column, among the 4 Remove-arm rows only.
    reg_rank, cost_rank = {}, {}     # (task, key) -> rank in {1,2}
    for t in tabs:
        tk = t["task"]
        rv = [t["col"].get(k, {}).get("regret_mean", float("nan"))
              for k in REMOVE_KEYS]
        cv = [t["col"].get(k, {}).get("cost_mean", float("nan"))
              for k in REMOVE_KEYS]
        for i, r in _rank_largest(rv).items():
            reg_rank[(tk, REMOVE_KEYS[i])] = r
        for i, r in _rank_largest(cv).items():
            cost_rank[(tk, REMOVE_KEYS[i])] = r

    def full_row():
        cells = []
        for t in tabs:
            c = t["col"]["__full__"]
            cells.append(rf"& {cell(c['regret_mean'], regret_dec)} "
                         rf"& {cell(c['cost_mean'], cost_dec)} ")
        return r"Full HybridBO           " + "".join(cells).rstrip() + r" \\"

    def variant_row(key, vlabel):
        cells = []
        for t in tabs:
            tk = t["task"]
            c = t["col"].get(key)
            if c is None or c["n"] == 0:
                cells.append("& -- & -- ")
                continue
            is_remove = key in REMOVE_KEYS
            rrank = reg_rank.get((tk, key)) if is_remove else None
            crank = cost_rank.get((tk, key)) if is_remove else None
            cells.append(
                rf"& {cell(c['regret_mean'], regret_dec, full_reg[tk], rrank)} "
                rf"& {cell(c['cost_mean'], cost_dec, full_cost[tk], crank)} ")
        return f"{vlabel:<24} " + "".join(cells).rstrip() + r" \\"

    L.append(full_row())
    L.append(r"\midrule")
    for vlabel, key, note in VARIANTS:
        if key is None:
            L.append(r"\midrule")
            continue
        L.append(variant_row(key, vlabel))

    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(L)


def write_csv(tabs, path):
    keys = ["task", "variant", "n", "regret_mean", "regret_std",
            "cost_mean", "cost_std", "fstar", "lam"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for t in tabs:
            for key, c in t["col"].items():
                w.writerow(dict(task=t["task"], variant=c["label"],
                                n=c["n"], regret_mean=c["regret_mean"],
                                regret_std=c["regret_std"],
                                cost_mean=c["cost_mean"],
                                cost_std=c["cost_std"],
                                fstar=t["fstar"], lam=t["lam"]))


def make_caption(tabs):
    ns = sorted({c["n"] for t in tabs for c in t["col"].values()
                 if c["n"] > 0})
    nstr = (f"$n{{=}}{ns[0]}$" if len(ns) == 1
            else f"$n\\in[{ns[0]},{ns[-1]}]$")
    return (rf"Ablation of HybridBO components across {len(tabs)} benchmarks, "
            rf"mean over {nstr} seeds. Each variant reports regret $R$ and "
            rf"cumulative LLM cost $C$ (CNY). Regret uses the same $f^\star$ "
            rf"anchor as Table~\ref{{tab:cost-performance-deepseek-try}}, "
            rf"pooled over both backends.")


CAPTION = None      # built dynamically in main() unless --caption overrides
LABEL = "tab:ablation-table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--fstar-csv", default=FSTAR_CSV)
    ap.add_argument("--outdir", type=Path, default=OUT_TAB)
    ap.add_argument("--regret-dec", type=int, default=4)
    ap.add_argument("--cost-dec", type=int, default=5)
    ap.add_argument("--caption", default=None,
                    help="override the auto-generated caption")
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    if args.fstar_csv:
        _load_fstar_csv(args.fstar_csv)

    tasks = args.tasks or TASKS
    args.outdir.mkdir(parents=True, exist_ok=True)
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Full     <- {MAIN_ROOT}/raw/")
    print(f"Variants <- {ABL_ROOT}/raw/")
    if args.fstar_csv:
        src = f"CSV {args.fstar_csv}"
    elif STRICT_REFERENCE:
        src = "hardcoded FSTAR_REFERENCE (strict)"
    else:
        src = "hardcoded FSTAR_REFERENCE, recompute-fallback"
    print(f"f* anchor: {src}")
    print(f"Tabulating backend: {MODEL_TABULATED}")
    print(f"{'-' * 60}")
    print(f"  {'task':24} {'f* used':>18}")
    for t in (args.tasks or TASKS):
        fs = (_FSTAR_FROM_CSV.get(t) if _FSTAR_FROM_CSV else None)
        fs = fs if fs is not None else FSTAR_REFERENCE.get(t, "(recompute)")
        print(f"  {t:24} {str(fs):>18}")
    print(f"{'-' * 60}\n")

    logpath = OUT_LOG / f"ablation_shared_{ts}.txt"
    with open(logpath, "w") as log:
        log.write(f"Ablation with shared f* -- {ts}\n")
        tabs = []
        for task in tasks:
            print(f"=== {task} ===")
            tab = build_task(task, log)
            if tab:
                tabs.append(tab)
                log.write(f"  {task}: f*={tab['fstar']}, lambda={tab['lam']}\n")

    if not tabs:
        print("No tasks built -- nothing written.")
        return

    caption = args.caption or make_caption(tabs)
    tex = render(tabs, caption, LABEL, args.regret_dec, args.cost_dec)
    tex_path = args.outdir / f"ablation_all_{ts}.tex"
    tex_path.write_text(tex + "\n")
    csv_path = args.outdir / f"ablation_all_{ts}.csv"
    write_csv(tabs, csv_path)

    print(f"\n-> {tex_path}")
    print(f"-> {csv_path}")
    print(f"-> {logpath}")
    print("\nPreamble needs: \\usepackage{booktabs}")
    if args.show:
        print("\n" + tex)


if __name__ == "__main__":
    main()