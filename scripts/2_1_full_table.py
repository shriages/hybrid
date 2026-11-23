#!/usr/bin/env python3
"""
Regret + cost table for ONE LLM backend (DeepSeek), with a shared f* anchor.

DESIGN
------
1. TASKS is the eight requested tasks.
2. ONE backend is tabulated (`MODEL`), so the table is a single flat CSV with
   no per-backend columns and therefore no blank cells. The f* anchor is a
   SEPARATE setting (`FSTAR_MODELS`) that pools over BOTH DeepSeek and GPT:
   regret is measured against the best value either backend reached, so these
   numbers stay on the same scale as a GPT table built from the same anchor.
   Setting FSTAR_MODELS = [MODEL] would rescale every row and break that.
3. Baselines are gp, tpe, smac, llm, lgbo, llambo. Online rows are lambda=10
   over three arm pools, each stored in ONE directory form and read by its own
   code path (see ONLINE_GROUPS):
        gp-tpe-llm-lgbo-llambo   <- _arms_ suffix
        tpe-llm-lgbo-llambo      <- _arms_ suffix
        gp-llm-lgbo-llambo       <- no suffix
   Per task: the six baselines first, then the three pools.
4. Columns: task | name | kind | lam | arms | fstar
            | regret_mean | regret_std | cost_mean | cost_std | n

MISSING CELLS
-------------
Some (task, baseline) pairs have no runs by design -- smac cannot be
implemented for every task. An absent baseline is excluded from the seed
intersection rather than voiding it (otherwise ONE missing baseline silently
drops the whole task, online rows included) and simply gets no row. Every
such gap is listed in coverage_<ts>.csv so it is never mistaken for an
oversight.

Arm pools are matched by SET, not by string, so a directory written
`gp-llm-llambo-lgbo` matches a pool declared in any other order.

Run from the project root (2_HybridBO/hybrid_bo):
    python scripts/21_cost_performance_tpe.py

Output:
    results/logs/tables_<ts>.csv
    results/logs/coverage_<ts>.csv
    results/logs/fstar_<ts>.csv
    results/logs/seeds_<ts>.csv
"""
import os
import re
import sys
import csv
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Project root & imports ──────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parent.parent      # .../2_HybridBO/hybrid_bo
os.chdir(PKG_ROOT)
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hybrid_bo.experiments.result_store import ResultStore

RESULTS_DIR = PKG_ROOT / "results"
LOG_DIR     = RESULTS_DIR / "logs"
store = ResultStore(root_dir=str(RESULTS_DIR))

# ── Backends ────────────────────────────────────────────────────────────────
# label -> model tag on disk. Edit the tags to match your result dirs.
# ── Backends ────────────────────────────────────────────────────────────────
# ONE backend is tabulated. The output is a single DeepSeek table; there is no
# per-backend pivot, so no column can be blank for want of a run.
# MODEL = "deepseek-v4-flash"
MODEL = "gpt-5.4-mini"

# The f* anchor still pools over BOTH backends. This is deliberate and is the
# reason MODEL and FSTAR_MODELS are separate settings: regret is measured
# against the best value ANY backend reached, so the DeepSeek numbers here
# remain on the same scale as a GPT table produced later from the same anchor.
# Narrowing this to [MODEL] would rescale every row.
FSTAR_MODELS = ["deepseek-v4-flash", "gpt-5.4-mini"]

MAIN_ALPHA  = 0.01
MAIN_ETA    = 7
MAIN_LAMBDA = 10.0          # every online row is lambda=10

BASELINES = ["gp", "tpe", "smac", "llm", "lgbo", "llambo"]

# ── Online groups ───────────────────────────────────────────────────────────
# Each arm pool is stored on disk in exactly ONE directory form, and the two
# forms are read by different code paths. Declaring the form per pool keeps
# them from being mixed:
#
#   "suffixed"  -> online_hybrid_lambda_10.00_alpha_0.01_eta_7.00_arms_<pool>
#   "bare"      -> online_hybrid_lambda_10.00_alpha_0.01_eta_7.00
#
# A pool declared "suffixed" will NEVER match a bare directory, and a pool
# declared "bare" will NEVER match a suffixed one -- there is no fallback
# between the two. Exactly one pool may be "bare", since a bare dirname
# carries no arm information to tell two of them apart.
#
# Order here is the group order in the table.
ONLINE_GROUPS = [
    (frozenset({"gp", "tpe", "llm", "lgbo", "llambo"}), "suffixed"),
    (frozenset({"tpe", "llm", "lgbo", "llambo"}),       "suffixed"),
    (frozenset({"gp", "llm", "lgbo", "llambo"}),        "bare"),
]

# Derived lookups. BARE_POOL is the pool a bare dirname resolves to.
_SUFFIXED_POOLS = {p for p, form in ONLINE_GROUPS if form == "suffixed"}
_BARE_POOLS     = [p for p, form in ONLINE_GROUPS if form == "bare"]
if len(_BARE_POOLS) > 1:
    raise ValueError(
        "At most one arm pool can use the bare (no _arms_) directory form; "
        f"got {[sorted(p) for p in _BARE_POOLS]}. A bare dirname carries no "
        "arm information, so two bare pools are indistinguishable on disk."
    )
BARE_POOL = _BARE_POOLS[0] if _BARE_POOLS else None


def pool_label(pool):
    """Stable display label for an arm-pool set."""
    order = ["gp", "tpe", "llm", "lgbo", "llambo"]
    return "-".join([a for a in order if a in pool]
                    + sorted(pool - set(order)))


def group_label(pool):
    return pool_label(pool)


PAIR_SEEDS_ACROSS_ARM_POOLS = False

TASKS = [
    "lasso_breast_cancer",
    "yahpo_lcbench_7593",
    "hartmann6",
    "schwefel",
    "bayesmark_SVM_digits",
    "bayesmark_SVM_breast",
    "lasso_diabetes",
    "olympus_suzuki",
]

# ── Discovery ───────────────────────────────────────────────────────────────
# Two separate patterns, deliberately not one regex with an optional group:
# each read-in path matches only its own directory form.
_ONLINE_SUFFIXED_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)"
    r"_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)"
    r"_arms_(?P<arms>[\w-]+)$"
)
_ONLINE_BARE_RE = re.compile(
    r"^online_hybrid_lambda_(?P<lam>[\d.]+)"
    r"_alpha_(?P<alpha>[\d.]+)_eta_(?P<eta>[\d.]+)$"
)


def _raw_task_dir(task):
    return RESULTS_DIR / "raw" / task


def _match_ok(m, alpha, eta, lam):
    return (abs(float(m["alpha"]) - alpha) < 1e-9
            and abs(float(m["eta"]) - eta) < 1e-9
            and (lam is None or abs(float(m["lam"]) - lam) < 1e-9))


def discover_suffixed(task, alpha=MAIN_ALPHA, eta=MAIN_ETA, lam=MAIN_LAMBDA):
    """READ-IN A: directories carrying an _arms_ suffix.

    The pool is parsed straight out of the dirname and compared as a SET, so
    `gp-llm-llambo-lgbo` matches a pool declared in any other order. Bare
    directories are invisible to this function.
    """
    d = _raw_task_dir(task)
    if not d.exists():
        return []
    out = []
    for sub in d.iterdir():
        if not sub.is_dir():
            continue
        m = _ONLINE_SUFFIXED_RE.match(sub.name)
        if not m or not _match_ok(m, alpha, eta, lam):
            continue
        pool = frozenset(m["arms"].split("-"))
        if pool not in _SUFFIXED_POOLS:
            continue
        out.append((round(float(m["lam"]), 4), pool, sub.name))
    return out


def discover_bare(task, alpha=MAIN_ALPHA, eta=MAIN_ETA, lam=MAIN_LAMBDA):
    """READ-IN B: directories with NO _arms_ suffix.

    The dirname says nothing about arms, so the pool comes from BARE_POOL --
    declared in ONLINE_GROUPS, not inferred. Suffixed directories are
    invisible to this function.
    """
    if BARE_POOL is None:
        return []
    d = _raw_task_dir(task)
    if not d.exists():
        return []
    out = []
    for sub in d.iterdir():
        if not sub.is_dir():
            continue
        m = _ONLINE_BARE_RE.match(sub.name)
        if not m or not _match_ok(m, alpha, eta, lam):
            continue
        out.append((round(float(m["lam"]), 4), BARE_POOL, sub.name))
    return out


def discover_main_configs(task, alpha=MAIN_ALPHA, eta=MAIN_ETA,
                          lam=MAIN_LAMBDA):
    """Union of both read-in paths, ordered by ONLINE_GROUPS then lambda.

    Returns [(lam, pool_frozenset, dirname), ...]. Because each pool is served
    by exactly one read-in path, a pool can never pick up rows from both.
    """
    configs = (discover_suffixed(task, alpha, eta, lam)
               + discover_bare(task, alpha, eta, lam))
    rank = {p: i for i, (p, _) in enumerate(ONLINE_GROUPS)}
    return sorted(configs, key=lambda r: (rank.get(r[1], 99), r[0]))
def online_groups(configs):
    """Distinct arm pools present, in discovery order."""
    seen = []
    for _, key, _ in configs:
        if key not in seen:
            seen.append(key)
    return seen


def _seeds_in_dir(method_dir: Path, model: str):
    if not method_dir.exists():
        return set()
    seeds = set()
    for p in method_dir.glob(f"{model}/seed_*.pkl"):
        m = re.match(r"seed_(\d+)\.pkl", p.name)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def baselines_with_runs(task, model):
    """Baselines that actually have runs for this (task, model).

    A baseline with no runs must NOT veto the seed intersection. Previously an
    empty set made `all(sets)` false and the function returned [], which
    silently dropped every row for the task -- online rows included -- with no
    error. smac/yahpo_lcbench_7593 is exactly that case.
    """
    d = _raw_task_dir(task)
    return [b for b in BASELINES if _seeds_in_dir(d / b, model)]


def discover_seeds_intersection(task, dirname, model, include_baselines=True):
    """Seeds shared by this run dir and every baseline THAT HAS RUNS.

    Baselines with no runs are excluded from the intersection rather than
    voiding it; the gap is reported to stdout instead, so it stays visible
    rather than silently emptying the table.
    """
    d = _raw_task_dir(task)
    own = _seeds_in_dir(d / dirname, model)
    if not own:
        return []
    sets = [own]
    if include_baselines:
        for opt in baselines_with_runs(task, model):
            sets.append(_seeds_in_dir(d / opt, model))
    return sorted(set.intersection(*sets)) if sets else []


def seed_sets_for_task(task, configs, model):
    """(all_seeds, {group_key: [seeds]}) for one backend."""
    all_seeds = sorted(set().union(*[
        set(discover_seeds_intersection(task, dn, model,
                                        include_baselines=False))
        for _, _, dn in configs
    ])) if configs else []
    if not all_seeds:
        all_seeds = list(range(10))

    groups = online_groups(configs)

    if PAIR_SEEDS_ACROSS_ARM_POOLS:
        per_cfg = [set(discover_seeds_intersection(task, dn, model))
                   for _, _, dn in configs]
        common = sorted(set.intersection(*per_cfg)) if per_cfg else []
        if not common:
            common = all_seeds
            print(f"[warn] {task}/{model}: no seeds shared by all online "
                  f"groups and all baselines; falling back to {common}")
        return all_seeds, {g: common for g in groups}

    common_by_group = {}
    for g in groups:
        per_cfg = [set(discover_seeds_intersection(task, dn, model))
                   for _, k, dn in configs if k == g]
        common = sorted(set.intersection(*per_cfg)) if per_cfg else []
        if not common:
            group_seeds = sorted(set().union(*[
                set(discover_seeds_intersection(task, dn, model,
                                                include_baselines=False))
                for _, k, dn in configs if k == g
            ])) if per_cfg else []
            common = group_seeds or all_seeds
            print(f"[warn] {task}/{model} arms={group_label(g)}: no seeds "
                  f"shared with all baselines; falling back to {common}")
        common_by_group[g] = common
    return all_seeds, common_by_group


# ── Loaders ─────────────────────────────────────────────────────────────────
def load_runs_for_model(task, opt_name, seeds, model):
    out = []
    for s in seeds:
        try:
            if store.exists(task, opt_name, model, s):
                out.append(store.load(task, opt_name, model, s))
        except Exception:
            pass
    return out


# ── Regret reference (f*), SHARED across backends ───────────────────────────
_FSTAR_CACHE = {}


def task_fstar(task):
    try:
        from hybrid_bo.experiments.online_runner import OnlineExperimentRunner
        t = OnlineExperimentRunner._make_task(task)
        fs = t.f_star()
        return None if fs is None else float(fs)
    except Exception:
        return None


def fstar_for(task, all_seeds):
    """Exact optimum if the task exposes one, else best-found value pooled
    over EVERY backend in FSTAR_MODELS and EVERY arm pool. Cached per task, so
    both backend tables are anchored on the identical reference."""
    if task in _FSTAR_CACHE:
        return _FSTAR_CACHE[task]
    fs = task_fstar(task)
    if fs is None:
        best = -np.inf
        configs = discover_main_configs(task)
        for model in FSTAR_MODELS:
            for _, _, dirname in configs:
                for r in load_runs_for_model(task, dirname, all_seeds, model):
                    best = max(best, r.f_best)
            for opt in BASELINES:
                for r in load_runs_for_model(task, opt, all_seeds, model):
                    best = max(best, r.f_best)
        fs = best if np.isfinite(best) else None
    _FSTAR_CACHE[task] = fs
    return fs


def regret(f_best_value, task, all_seeds):
    fs = fstar_for(task, all_seeds)
    return np.nan if fs is None else fs - f_best_value


# ── Row builder ─────────────────────────────────────────────────────────────
def build_row(task, name, kind, results, all_seeds, arms="", lam=""):
    if not results:
        return None
    reg  = np.array([regret(r.f_best, task, all_seeds) for r in results],
                    dtype=float)
    cost = np.array([r.total_cost for r in results], dtype=float)
    fs = fstar_for(task, all_seeds)
    return dict(
        task=task,
        name=name,
        kind=kind,
        lam=lam,
        arms=arms,
        fstar=("" if fs is None else f"{fs:.12g}"),
        regret_mean=float(np.nanmean(reg)),
        regret_std=float(np.nanstd(reg)),
        cost_mean=float(np.nanmean(cost)),
        cost_std=float(np.nanstd(cost)),
        n=len(results),
    )


# ── Table builder ───────────────────────────────────────────────────────────
def build_rows():
    """Grouped rows for MODEL: per task -> 6 baselines, then 3 online pools."""
    rows, fstar_report, seed_report = [], [], {}

    for task in TASKS:
        configs = discover_main_configs(task)
        if not configs:
            print(f"[skip] {task}: no online results matching the requested "
                  f"lambda={MAIN_LAMBDA:g} groups.")
            continue

        all_seeds, common_by_group = seed_sets_for_task(task, configs, MODEL)
        groups = online_groups(configs)

        print(f"=== {task} | groups={[group_label(g) for g in groups]} ===")
        for g in groups:
            print(f"      arms={group_label(g):28s} "
                  f"seeds={common_by_group[g]}")
            seed_report[f"{task}@{group_label(g)}"] = list(common_by_group[g])

        fstar_report.append((
            task,
            "exact" if task_fstar(task) is not None
            else f"pooled({len(FSTAR_MODELS)} backends)",
            fstar_for(task, all_seeds),
        ))

        # GROUP 1 — baselines. Seeds = union of the groups' paired seed sets,
        # so no baseline is scored on a seed no online run was paired against.
        baseline_seeds = sorted(set().union(*[
            set(s) for s in common_by_group.values()
        ])) if common_by_group else []

        # A baseline with no runs is EXPECTED for some (task, baseline) pairs:
        # smac cannot be implemented for every task, so it is absent by design
        # rather than by accident. It is excluded from the seed intersection
        # (otherwise the task would report nothing at all) and simply has no
        # row. Recorded in the coverage export so the gap is explicit.
        absent = [b for b in BASELINES
                  if not _seeds_in_dir(_raw_task_dir(task) / b, MODEL)]
        if absent:
            print(f"  [note] {task}: no runs for {absent} -- excluded from "
                  f"seed pairing; no row emitted for them.")
        for opt in BASELINES:
            runs = load_runs_for_model(task, opt, baseline_seeds, MODEL)
            row = build_row(task, opt, "baseline", runs, all_seeds)
            if row:
                rows.append(row)

        # GROUPS 2..k — one per arm pool, in ONLINE_GROUPS order. Each pool is
        # read by exactly one path, so no pool mixes the two directory forms.
        for lam, pool, dirname in configs:
            seeds = common_by_group[pool]
            runs = load_runs_for_model(task, dirname, seeds, MODEL)
            name = f"online(lambda={lam:g}, arms={pool_label(pool)})"
            row = build_row(task, name, "online", runs, all_seeds,
                            arms=pool_label(pool), lam=f"{lam:g}")
            if row:
                rows.append(row)
            else:
                print(f"  [warn] no runs for {dirname}")

    return rows, fstar_report, seed_report


KEYS = ["task", "name", "kind", "lam", "arms", "fstar",
        "regret_mean", "regret_std", "cost_mean", "cost_std", "n"]


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sort_rows(rows):
    """Task order from TASKS; within a task, baselines then online pools."""
    task_rank = {t: i for i, t in enumerate(TASKS)}
    base_rank = {b: i for i, b in enumerate(BASELINES)}
    pool_rank = {pool_label(p): i for i, (p, _) in enumerate(ONLINE_GROUPS)}

    def key(r):
        if r["kind"] == "baseline":
            return (task_rank.get(r["task"], 99), 0,
                    base_rank.get(r["name"], 99))
        return (task_rank.get(r["task"], 99), 1,
                pool_rank.get(r["arms"], 99))

    return sorted(rows, key=key)


def preview(rows):
    print()
    print(f"{'task':22s} {'method':40s} {'regret':>19s} {'cost':>20s}  n")
    last = None
    for r in rows:
        if r["task"] != last:
            print("-" * 112)
            last = r["task"]
        print(f"{r['task']:22s} {r['name']:40s} "
              f"{r['regret_mean']:9.4f}±{r['regret_std']:<8.4f} "
              f"${r['cost_mean']:9.5f}±{r['cost_std']:<8.5f} {r['n']}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print(f"Tabulating model   : {MODEL}")
    print(f"Shared f* anchor   : {FSTAR_MODELS}")
    print("  (tasks with a known optimum use it exactly and ignore the pool)")
    print(f"Online rows        : lambda={MAIN_LAMBDA:g}, "
          f"alpha={MAIN_ALPHA}, eta={MAIN_ETA}")
    for _pool, _form in ONLINE_GROUPS:
        _tag = "_arms_ suffix" if _form == "suffixed" else "no suffix"
        print(f"    {pool_label(_pool):<26s} <- {_tag}")
    print(f"Baselines          : {BASELINES}")
    print(f"Seed pairing       : "
          f"{'task-wide' if PAIR_SEEDS_ACROSS_ARM_POOLS else 'per group'}")
    print("=" * 78)

    rows, fstar_report, seed_report = build_rows()
    if not rows:
        print("No rows produced -- nothing written.")
        return
    rows = sort_rows(rows)

    tables_csv = LOG_DIR / f"tables_{ts}.csv"
    write_csv(tables_csv, rows, KEYS)
    preview(rows)
    print(f"\nDONE -> {tables_csv.relative_to(PKG_ROOT)}  ({len(rows)} rows)")

    # ── Coverage: which (task, method) cells have no runs, and why ──────────
    # Emitted so an absent row is never mistaken for a missing table entry.
    # smac is absent on tasks where it cannot be implemented.
    cov_csv = LOG_DIR / f"coverage_{ts}.csv"
    have = {(r["task"], r["name"]) for r in rows}
    with open(cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "status"])
        for task in TASKS:
            for b in BASELINES:
                w.writerow([task, b,
                            "ok" if (task, b) in have else "no_runs"])
            for pool, _ in ONLINE_GROUPS:
                nm = f"online(lambda={MAIN_LAMBDA:g}, arms={pool_label(pool)})"
                w.writerow([task, nm,
                            "ok" if (task, nm) in have else "no_runs"])
    print(f"coverage       -> {cov_csv.relative_to(PKG_ROOT)}")

    # ── f* provenance ──────────────────────────────────────────────────────
    # Single source of truth for the regret anchor, pooled over FSTAR_MODELS.
    fstar_csv = LOG_DIR / f"fstar_{ts}.csv"
    with open(fstar_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "source", "fstar"])
        for task, src, fs in fstar_report:
            w.writerow([task, src, "" if fs is None else f"{fs:.12g}"])
    print(f"f* table       -> {fstar_csv.relative_to(PKG_ROOT)}")

    seeds_csv = LOG_DIR / f"seeds_{ts}.csv"
    with open(seeds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task@arms", "seeds"])
        for key, seeds in sorted(seed_report.items()):
            w.writerow([key, " ".join(str(s) for s in seeds)])
    print(f"seed table     -> {seeds_csv.relative_to(PKG_ROOT)}")

    if fstar_report:
        print(f"\n{'-' * 78}")
        print(f"f* reference used  (pool = {FSTAR_MODELS})")
        print(f"{'-' * 78}")
        print(f"  {'task':26s} {'source':22s} {'f*':>14s}")
        for task, src, fs in fstar_report:
            fs_s = "n/a" if fs is None else f"{fs:.6f}"
            print(f"  {task:26s} {src:22s} {fs_s:>14s}")


if __name__ == "__main__":
    main()