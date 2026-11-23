"""
One-shot regret + cost + t-test LaTeX table, built DIRECTLY from
21_cost_performance_tpe.py's output -- no t1/t2 intermediate CSVs.

    INPUT   results/logs/tables_<ts>.csv        (from 21_cost_performance_tpe.py)
    OUTPUT  results/logs/t3_full_<ts>.tex

WHY DIRECT
----------
22_make_tables.py / 23_render_latex.py / 25_render_ttest_split.py split
compute (22) from render (23/25) via t1_cost_perf_<ts>.csv and
t2_ttest_<ts>.csv. This script folds both steps back together and reads
tables_<ts>.csv straight off disk, because the deliverable now needs a
column (Cost) that lives in table 1's computation but a layout (per-task
Regret/t/p triplets, split into stacked table* parts) that lives in table
2's renderer. Keeping two intermediate CSVs in sync for one new column
wasn't worth it; the Welch/Holm logic is copied verbatim from
22_make_tables.py instead.

LAYOUT
------
Same split strategy as 25_render_ttest_split.py: no \resizebox, 10pt roman,
tasks stacked into table* parts of PER_ROW benchmarks each, sharing one
caption via \ContinuedFloat. Per task there are THREE sub-columns:
Regret, Cost, and a combined $t\,(p)$ column (the $t$ and $p$ values used
to render as two separate columns; they're now merged into one cell as
"t (p)" to cut 4 sub-columns/task down to 3 and keep the two-part table
narrower without dropping any number).

    Regret -- scaled per task ($\times 10^e$), same scale_exponent() logic
              as 25_render_ttest_split.py. NOW ALSO RANKED like Cost: bold
              = lowest regret, underline = second-lowest, ranked PER TASK
              across the online reference row AND every baseline, dense
              ranking (ties share a rank, so the next distinct value still
              gets the underline).
    Cost   -- CNY, fixed 3dp, exact zero prints as '0'. Bold = cheapest,
              underline = second cheapest, ranked PER TASK across the
              online reference row AND every baseline, with zero-cost
              methods (gp/tpe/smac make no API calls) competing for the
              rank -- i.e. COST_RANK_INCLUDES_ZERO=True from
              23_cost_performance_latex.py, so GP/TPE/SMAC take the bold
              0 rather than being excluded.
    t (p)  -- Welch's unpaired t-test (reference vs. each baseline), plus
              Holm correction within each task's family of baseline tests.
              t and p are rendered together as one cell, "$t\,(p)$";
              bold/underline is applied to the whole cell based on the p
              value only (same threshold/logic as before). No --seeds
              option here (unlike 22_make_tables.py) since this script
              reads only the summary CSV; add one the same way if a paired
              test is ever needed.

Both rank columns (Regret, Cost) use the SAME dense rank_map_dense() helper
and the SAME mark() wrapper as before -- only the values fed into them
differ, so a rank-1/rank-2 regret and a rank-1/rank-2 cost read identically
on the page.

Usage
-----
    python scripts/26_full_cost_ttest_latex.py
    python scripts/26_full_cost_ttest_latex.py --print
    python scripts/26_full_cost_ttest_latex.py --per-row 4
    python scripts/26_full_cost_ttest_latex.py --sig-col p_holm
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TABLES_CSV = "output/file/of/2_1_full_table.csv"


LOG_DIR = Path("results/logs")

# Arm pool of the online run used as the reference in the t-tests.
DEFAULT_REFERENCE = "gp-llm-lgbo-llambo"

BASELINE_ORDER = ["gp", "tpe", "smac", "llm", "llambo", "lgbo"]
ONLINE_LABEL = "HybridBO"

ALPHA = 0.05

# Task order, and which PER_ROW share a row. Part (a) is the first PER_ROW
# entries, part (b) the rest. Empty list = CSV order. Matches 21_...py's
# TASKS (8 tasks) so a "2 x 4" split falls out of the default PER_ROW=4.
TASKS = [
    # part (a)
    "hartmann6",
    "schwefel",
    "bayesmark_SVM_breast",
    "olympus_suzuki",
    # part (b)
    "lasso_breast_cancer",
    "yahpo_lcbench_7593",
    "bayesmark_SVM_digits",
    "lasso_diabetes",
]

PER_ROW = 4                     # benchmarks per table* part -> 2 parts of 4

PRETTY = {"gp": "GP", "tpe": "TPE", "smac": "SMAC",
          "llm": "LLM", "lgbo": "LGBO", "llambo": "LLAMBO"}

TASK_PRETTY = {
    "lasso_breast_cancer": "Lasso Breast",
    "yahpo_lcbench_7593": "LCBench 7593",
    "hartmann6": "Hartmann6",
    "schwefel": "Schwefel",
    "bayesmark_SVM_digits": "SVM Digits",
    "bayesmark_SVM_breast": "SVM Breast",
    "lasso_diabetes": "Lasso Diabetes",
    "olympus_suzuki": "Suzuki",
}


def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
                  .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def pm(m):
    return PRETTY.get(m, esc(m))


def pt(t):
    return TASK_PRETTY.get(t, esc(t))


# ── IO: read tables_<ts>.csv directly, same schema 21_...py writes ─────────
def read_summary(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("regret_mean", "regret_std", "cost_mean", "cost_std",
                  "fstar"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
        r["n"] = int(r["n"])
    return rows


def method_key(row):
    """Short label: the baseline name, or the arm pool for an online row."""
    return row["name"] if row["kind"] == "baseline" else row["arms"]


def resolve(cli, const, what):
    p = Path(cli or const)
    if not p.is_file():
        sys.exit(f"{what} not found: {p}\nEdit TABLES_CSV at the top, or "
                 f"pass the path on the command line.")
    return p


def tasks_in_order(rows, only=None):
    seen = []
    for r in rows:
        if r["task"] not in seen:
            seen.append(r["task"])
    if only:
        missing = [t for t in only if t not in seen]
        if missing:
            sys.exit(f"Task(s) not in CSV: {missing}\nAvailable: {seen}")
        return list(only)
    return seen


def fnum(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


# ── Stats: Welch t-test + Holm, copied from 22_make_tables.py ──────────────
def welch_from_summary(m1, s1, n1, m2, s2, n2, alternative="two-sided"):
    """Welch t-test from summary stats. scipy wants SAMPLE std; the summary
    table carries POPULATION std (ddof=0), so rescale by sqrt(n/(n-1))."""
    s1 = s1 * np.sqrt(n1 / (n1 - 1)) if n1 > 1 else s1
    s2 = s2 * np.sqrt(n2 / (n2 - 1)) if n2 > 1 else s2
    res = stats.ttest_ind_from_stats(
        mean1=m1, std1=s1, nobs1=n1,
        mean2=m2, std2=s2, nobs2=n2,
        equal_var=False, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [float("nan")] * len(pvals)
    m = len(idx)
    prev = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out


def rank_map_dense(values):
    """Dense 1-based ranks ascending; equal values (e.g. several 0-cost
    baselines) share a rank, so the next distinct value still gets rank 2
    and the underline doesn't silently disappear. Used for BOTH the Regret
    and the Cost column -- lower is better for both."""
    idx = [i for i, v in enumerate(values) if v == v]
    out, rank, prev = {}, 0, None
    for i in sorted(idx, key=lambda j: values[j]):
        if prev is None or abs(values[i] - prev) > 1e-15:
            rank += 1
            prev = values[i]
        out[i] = rank
    return out


def compute_task(rows, task, reference, alternative="two-sided"):
    """Everything needed to render one task's block: ref row, per-baseline
    Welch/Holm stats, and a regret rank + cost rank shared by ref + every
    baseline."""
    trows = [r for r in rows if r["task"] == task]
    by = {method_key(r): r for r in trows}
    if reference not in by:
        print(f"  [warn] {task}: reference '{reference}' absent, skipped")
        return None
    ref = by[reference]
    bases = [b for b in BASELINE_ORDER if b in by]
    if not bases:
        print(f"  [warn] {task}: no baselines present")

    pvals, stats_rows = [], []
    for b in bases:
        base = by[b]
        t, p = welch_from_summary(
            ref["regret_mean"], ref["regret_std"], ref["n"],
            base["regret_mean"], base["regret_std"], base["n"], alternative)
        stats_rows.append(dict(baseline=b, t=t, p=p))
        pvals.append(p)
    for r, ph in zip(stats_rows, holm(pvals)):
        r["p_holm"] = ph
    stats_by_baseline = {r["baseline"]: r for r in stats_rows}

    # Rank: dense, PER TASK, across the reference row + every present
    # baseline. Cost includes zero-cost methods in the ranking
    # (COST_RANK_INCLUDES_ZERO=True); regret has no such exclusion since
    # regret is never trivially zero the way cost is.
    names = [reference] + bases
    costs = [ref["cost_mean"]] + [by[b]["cost_mean"] for b in bases]
    regrets = [ref["regret_mean"]] + [by[b]["regret_mean"] for b in bases]
    cr = rank_map_dense(costs)
    rr = rank_map_dense(regrets)
    cost_rank = {names[i]: cr.get(i) for i in range(len(names))}
    regret_rank = {names[i]: rr.get(i) for i in range(len(names))}

    return dict(ref=ref, by=by, bases=bases,
               stats_by_baseline=stats_by_baseline,
               cost_rank=cost_rank, regret_rank=regret_rank)


# ── Numbers (regret scaling + compact p, from 25_render_ttest_split.py) ────
def scale_exponent(values):
    nz = [abs(v) for v in values if v == v and v != 0]
    if not nz:
        return 0
    e = int(math.floor(math.log10(max(nz))))
    return e if abs(e) >= 2 else 0


def fscaled(v, e, dec=2):
    if v != v:
        return "--"
    return f"{(v / (10.0 ** e)):.{dec}f}"


def fcost(v, dec=3):
    """Fixed-decimal CNY, as in 23_cost_performance_latex.py. Exact zero
    stays '0' rather than '0.000', so the bold 0 reads cleanly."""
    if v != v:
        return "--"
    if v == 0:
        return "0"
    return f"{v:.{dec}f}"


def ft(v):
    return "--" if v != v else f"{v:.2f}"


def fp(v):
    """Compact p: 3 decimals, else 1-sig-fig scientific as 4e-11."""
    if v != v:
        return "--"
    if v >= 1e-3:
        return f"{v:.3f}"
    mant, exp = f"{v:.0E}".split("E")
    return f"{int(float(mant))}e{int(exp)}"


def ftp(t, p):
    """Combined 't (p)' cell, e.g. '-5.36\\,(8e\\text{-}5)'. Replaces the
    old separate $t$/$p$ columns; the whole cell is later wrapped in
    bold/underline based on p alone, same threshold as before. The t
    value itself gets a \\boxed{} outline whenever t > 0 (positive t
    means the reference beat the baseline, in the alternative="two-sided"
    / "greater" convention used here)."""
    tstr = ft(t)
    pstr = fp(p)
    if tstr == "--" and pstr == "--":
        return "--"
    if "e" in pstr:
        mant, exp = pstr.split("e")
        if exp.startswith("-"):
            exp = r"\text{-}" + exp[1:]
        pstr = f"{mant}e{exp}"
    if t == t and t > 0 and tstr != "--":
        tstr = rf"\boxed{{{tstr}}}"
    return rf"{tstr}\,({pstr})"


def bold(s):
    return "--" if s == "--" else rf"$\mathbf{{{s}}}$"


def under(s):
    return "--" if s == "--" else rf"\underline{{${s}$}}"


def math_(s):
    return s if s == "--" else f"${s}$"


def mark(s, rank):
    if rank == 1:
        return bold(s)
    if rank == 2:
        return under(s)
    return math_(s)


def header_label(task, e):
    if e == 0:
        return pt(task)
    return rf"{pt(task)} ($\times 10^{{{e}}}$)"


# ── Render one table* part (a "row" of PER_ROW benchmarks) ─────────────────
def render_part(rows, tasks, reference, sig_col, caption, label,
                continued, standalone_caption):
    task_blocks = {t: compute_task(rows, t, reference) for t in tasks}
    tasks = [t for t in tasks if task_blocks[t] is not None]
    ntask = len(tasks)

    present = set()
    for t in tasks:
        present |= set(task_blocks[t]["bases"])
    bases = [b for b in BASELINE_ORDER if b in present]

    exps = {}
    for t in tasks:
        blk = task_blocks[t]
        vals = [blk["ref"]["regret_mean"]] + \
               [blk["by"][b]["regret_mean"] for b in blk["bases"]]
        exps[t] = scale_exponent(vals)

    # NOTE: 3 sub-columns/task now (Regret, Cost, t(p)) instead of 4.
    L = [r"\begin{table*}[t]"]
    if continued:
        L.append(r"\ContinuedFloat")
    L += [r"\centering",
          r"\footnotesize",
          r"\setlength{\tabcolsep}{0.9mm}",
          rf"\begin{{tabular}}{{l {' '.join(['rrr'] * ntask)}}}",
          r"\toprule"]

    L.append("".join(rf"& \multicolumn{{3}}{{c}}{{\large {header_label(t, exps[t])}}} "
                     for t in tasks).rstrip() + r" \\")
    L.append("".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                     for i in range(ntask)))
    L.append("Baselines " +
             (r"& Regret & Cost & \multicolumn{1}{c}{$t\,(p)$} " * ntask) +
             r"\\")
    L.append(r"\midrule")

    ref_cells = []
    for t in tasks:
        blk = task_blocks[t]
        ref = blk["ref"]
        rrank = blk["regret_rank"].get(reference)
        crank = blk["cost_rank"].get(reference)
        ref_cells.append(
            rf"& {mark(fscaled(ref['regret_mean'], exps[t]), rrank)} & "
            rf"{mark(fcost(ref['cost_mean']), crank)} & -- ")
    L.append(rf"{ONLINE_LABEL} " + "".join(ref_cells).rstrip() + r" \\")
    L.append(r"\midrule")

    for b in bases:
        cells = []
        for t in tasks:
            blk = task_blocks[t]
            if b not in blk["by"]:
                cells.append("& -- & -- & -- ")            # e.g. SMAC absent
                continue
            base = blk["by"][b]
            st = blk["stats_by_baseline"].get(b)
            p = st[sig_col] if st else float("nan")
            tval = st["t"] if st else float("nan")
            tp_cell = ftp(tval, p)
            tp_tex = bold(tp_cell) if p == p and p < ALPHA else under(tp_cell)
            rrank = blk["regret_rank"].get(b)
            crank = blk["cost_rank"].get(b)
            cells.append(
                rf"& {mark(fscaled(base['regret_mean'], exps[t]), rrank)} & "
                rf"{mark(fcost(base['cost_mean']), crank)} & "
                rf"{tp_tex} ")
        L.append(f"{pm(b):<7} " + "".join(cells).rstrip() + r" \\")

    L += [r"\bottomrule", r"\end{tabular}"]
    L.append(rf"\caption{{{standalone_caption}}}")
    if label:
        L.append(rf"\label{{{label}}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


CAPTION = (r"Trial run: comparison of HybridBO against each baseline, "
           r"including the mean cumulative LLM cost (CNY). Bold marks "
           r"$p<0.05$ in the $t\,(p)$ column, and the lowest value in the "
           r"Regret and Cost columns; underline marks no significant "
           r"difference, or the second-lowest value in the Regret and "
           r"Cost columns. Regret values are means over 10 seeds, scaled "
           r"per benchmark as indicated in the header.")
LABEL = "tab:full-cost-ttest-openai-try"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="tables_<ts>.csv (default: TABLES_CSV constant)")
    ap.add_argument("--outdir", type=Path, default=LOG_DIR)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help=f"online arm pool to test (default {DEFAULT_REFERENCE})")
    ap.add_argument("--alternative", default="two-sided",
                    choices=["two-sided", "less", "greater"])
    ap.add_argument("--sig-col", default="p", choices=["p", "p_holm"])
    ap.add_argument("--caption", default=CAPTION)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--per-row", type=int, default=PER_ROW)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()

    src = resolve(args.input, TABLES_CSV, "tables_<ts>.csv")
    print(f"input: {src}")
    rows = read_summary(src)

    only = args.tasks if args.tasks is not None else (TASKS or None)
    tasks = tasks_in_order(rows, only)

    chunks = [tasks[i:i + args.per_row]
              for i in range(0, len(tasks), args.per_row)]
    parts = []
    for k, chunk in enumerate(chunks):
        first = (k == 0)
        letter = chr(ord("a") + k)
        if first:
            cap = rf"\textbf{{({letter})}} " + args.caption
        else:
            cap = rf"\textbf{{({letter})}} Continued."
        parts.append(render_part(
            rows, chunk, args.reference, args.sig_col, args.caption,
            args.label if first else f"{args.label}-{letter}",
            continued=not first, standalone_caption=cap))

    tex = "\n\n".join(parts)

    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    import re
    m = re.search(r"tables_(.+)$", src.stem)
    ts = m.group(1) if m else src.stem
    out = args.outdir / f"t3_full_{ts}.tex"
    out.write_text(tex)
    print(f"{src.name} -> {out}")
    print(f"{len(tasks)} benchmarks in {len(chunks)} parts of "
          f"{args.per_row}")
    print("\nPreamble needs: \\usepackage{booktabs}")
    if args.show:
        print("\n" + tex)


if __name__ == "__main__":
    main()