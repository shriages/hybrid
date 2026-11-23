# HybridBO — Reproducibility Supplement

This archive contains the source code and figure/table scripts for the paper's
experiments. It lets a reviewer (a) read exactly what the method does, (b) run
the method locally on a single task, and (c) regenerate the paper's figures and
tables from experiment outputs.

> **Data.** The full results are in `data/`. If you run the code and generate results, they are in `results/` (see "Data & results layout") and the scripts below
> will read them directly.


## 1. What HybridBO is

HybridBO is an online optimizer-switching supervisor for Bayesian optimization.
An Exp3 bandit chooses, per batch, among several BO "arms" -- a Gaussian-process
optimizer (`gp`), a plain LLM proposer (`llm`), and two LLM-assisted BO methods
(`llambo`, `lgbo`) -- trading optimization performance against the monetary cost
of LLM calls via a preference weight lambda. The thesis is *not* that LLM
assistance always helps, but that a supervisor can discover **when** it is worth
its cost.

The repository also implements the reported baselines and ablations: a TPE
baseline (`tpe`), a random-forest BO baseline (`smac`), fixed LLM->GP schedules
(`fixed_switch_*`), and the ablation variants (no-cost, no-exploration,
fixed-distribution, one-time-switch, arm-removal).


## 2. Repository layout

```
hybrid_bo/                       the installable package
  core/                          optimizer arms + the online supervisor
    online_hybrid_optimizer.py     the Exp3 switching loop
    online_run_result.py  run_result.py
    gp_optimizer.py  tpe_optimizer.py  smac_optimizer.py
    llm_optimizer.py llambo_optimizer.py lgbo_optimizer.py
  switching/exp3_selector.py     the bandit
  benchmarks/                    task definitions + base_task.py + task_metadata.py
    branin, hartmann6, rastrigin, schwefel, gallagher,
    bayesmark_task, yahpo_task, lasso_bench_task, olympus_dataset_task
  llm_backends/                  deepseek / openai / anthropic clients,
                                 pricing.py, cost_tracker.py, prompt_builder.py
  experiments/                   config, runner, online_runner, result_store,
                                 online_loader
  _*_compat.py                   small shims (configspace / sklearn / matplotlib)
scripts/                         figure and table generators (see Section 5)
modal/                           cloud orchestration for the full sweep
                                 (transparency only; NOT required to reproduce)
pyproject.toml                   dependencies and build config
```


## 3. Environments

Different benchmark families have **mutually incompatible dependencies**, so the
project uses two Python environments. This is intrinsic to the benchmarks, not a
packaging choice.

| Environment | Python | Covers | Notes |
|---|---|---|---|
| main  | 3.13 | the method; synthetic + bayesmark tasks | default |
| py310 | 3.10 | yahpo, lassobench, olympus tasks        | separate deps |

Dependencies are declared in `pyproject.toml`. Install the package and its deps
into the main environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .            # reads pyproject.toml
```

The `py310` environment is only needed for the yahpo / lassobench / olympus
tasks and has extra, version-pinned requirements that conflict with the main
env (see below). Create it with a Python 3.10 interpreter:

```bash
python3.10 -m venv .venv-py310
source .venv-py310/bin/activate
pip install -e .
# then the family-specific installs documented in "Known incompatibilities"
```

### Known dependency incompatibilities (documented, not bugs)

- **SMAC vs YAHPO.** `smac` (>=2.4) requires `ConfigSpace>=1.0`; `yahpo-gym==1.0.2`
  pins `ConfigSpace<=0.6.1`. They cannot coexist in one environment, so the
  `smac` baseline is evaluated on every benchmark **except** YAHPO.
- **YAHPO** needs `ConfigSpace==0.6.1`, `onnxruntime`, and the `yahpo_data`
  surrogate files (cloned separately); a small shim `_configspace_compat.py`
  bridges an API change.
- **LassoBench** is not on PyPI; install from source
  (`git clone https://github.com/ksehic/LassoBench && pip install -e .`) and add
  `setuptools<81` (its `hyperopt` dependency imports `pkg_resources`), plus
  `celer` and `sparse-ho`.
- **Olympus** emulators require `tensorflow==2.15.1`,
  `tensorflow-probability==0.23.0`, `tf-keras`, and `TF_USE_LEGACY_KERAS=1`.
- **SMAC** additionally needs a build toolchain (`swig`, `build-essential`) to
  compile `pyrfr`.

### API keys (only for *re-running* LLM methods)

The LLM arms call external APIs. To re-run `llm` / `llambo` / `lgbo` / the online
method, set the relevant key in your environment:

```bash
export DEEPSEEK_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

The LLM-free methods (`gp`, `tpe`, `smac`) need no key, and **every figure and
table can be regenerated from the provided results without any API key.**


## 4. Running the method locally (no cloud)

A single run is one `(task, optimizer, seed)`. The example below uses only
LLM-free components, so it needs no API key:

```bash
source .venv/bin/activate
python -m hybrid_bo.experiments.runner \
    --task branin --optimizer gp --seed 0 --T 61
```

Running the online supervisor (calls the LLM arms; needs a key and incurs cost):

```bash
export DEEPSEEK_API_KEY=...
python -m hybrid_bo.experiments.online_runner \
    --task branin --lambda 10 --seed 0 --T 61
```

Results are written under
`results/raw/<task>/<optimizer>/<model>/seed_<n>.pkl`. The optimizer directory
name encodes the online hyperparameters and, for non-default arm pools, the arm
roster, e.g. `online_hybrid_lambda_10.00_alpha_0.01_eta_7.00`.

> The full multi-task / multi-seed sweep was orchestrated on the cloud (see
> `modal/`), but any single run can be reproduced with the commands above. The
> cloud path is **not** required and depends on an account and secrets a reviewer
> would not have.


## 5. Scripts -> paper figures and tables

Run scripts from the repository root, e.g. `python scripts/1_PCA_trajectory.py`.
Each reads from `results/` and writes figures/tables to an output folder it
prints on start.

| Script | Produces |
|---|---|
| `1_PCA_trajectory.py`          | **Figure 2** (main) -- PCA trajectories |
| `2_1_full_table.py`            | the full results table (data for Table 2 (main) / Table 6) (appendix) |
| `2_6_full_table_latex.py`      | LaTeX of `2_1` output: **Table 2** (main) and **Table 6** (appendix) |
| `5_cost_performance.py`        | **Figure 3** (main) -- cost vs performance |
| `7_final_frontier.py`          | **Figure 4** (main) -- final cost-performance frontier |
| `8_0_half_histogram.py`        | **Figure 5** (main) -- half-split arm histograms |
| `9_circle_figure.py`           | **Figure 6** (main) |
| `16_detect_llm_suzuki.py`      | **Figure 7** (appendix) |
| `8_1_histogram_changes.py`     | **Figure 8** (appendix) -- histogram changes |
| `10_1_subset_arm_trajectory.py`| **Figures 9 and 13** (appendix) -- subset arm trajectories |
| `12_all_cost_performance.py`   | **Figure 10** (appendix) |
| `14_all_frontiers.py`          | **Figure 11** (appendix) -- all frontiers |
| `15_all_lambda_probability.py` | **Figure 12** (appendix) -- arm probability across lambda |
| `11_ablation_table.py`         | ablation table -- run *after* `modal/modal_ablation.py`, then summarize |

Ablation note: `11_ablation_table.py` reads the ablation runs produced by
`modal/modal_ablation.py` (saved under `results/ablation/`) and summarizes them
into the reported ablation table.


## 6. Data & results layout

Scripts expect results under `results/`, mirroring how runs are saved:

```
results/
  raw/<task>/<optimizer>/<model>/seed_<n>.pkl           main sweep + baselines
  ablation/raw/<task>/<variant>/<model>/seed_<n>.pkl    ablation runs
  figures/                                              script output
```

Unpack the separately-provided data upload into `results/` before running the
Section 5 scripts.


## 7. How performance and regret are defined

All tasks are cast as maximization (each task negates its objective as needed so
higher is always better); `f_best` is the best value observed in a run. Regret is
`f* - f_best`, with `f*` defined per benchmark family:

- **Synthetic** (branin, hartmann6, rastrigin, schwefel, gallagher): the exact
  known optimum.
- **LassoBench**: the benchmark's oracle loss (an absolute achievable floor).
- **Bayesmark / YAHPO / Olympus**: no closed-form optimum exists, so `f*` is the
  best value found by any method across all configurations (a shared "best-found"
  anchor). Regret on these tasks is therefore **pool-relative** -- distance to the
  best method, not to a true optimum -- and is comparable within, not across,
  benchmark families. The anchor is computed once and shared across all
  tables/figures so identical runs always report identical regret.


## 8. Reproducibility caveats

- **Pool-relative regret** on Bayesmark/YAHPO/Olympus (Section 7): read small
  gaps on these tasks with the shared-anchor definition in mind.
- **Task-side nondeterminism.** Some objectives are not perfectly deterministic
  for a fixed seed (sklearn internals in Bayesmark; neural-net emulators in
  Olympus). Synthetic tasks are exactly reproducible; LLM-free optimizers are
  otherwise reproducible from their seed. Reported numbers average over 10 seeds.
- **LLM output drift.** LLM-assisted arms depend on the backend's output
  formatting; parsers tolerate bare JSON, markdown-fenced JSON, and surrounding
  prose, but different providers/models can behave differently. Results are
  reported per backend.
- **SMAC excludes YAHPO** (Section 3).