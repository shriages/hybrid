"""
modal_baselines_extra.py — run the NEW baselines only.

Adds four baselines without touching modal_app.py or re-running anything you
already have:

    tpe                   Tree-structured Parzen Estimator (Optuna). No LLM,
                          costs $0. The "strong classical BO" reference.
    fixed_switch_llm      plain LLM  -> GP at tau = T // 2
    fixed_switch_llambo   LLAMBO     -> GP at tau = T // 2
    fixed_switch_lgbo     LGBO       -> GP at tau = T // 2

Results are saved into the SAME root as the existing baselines
(results/raw/<task>/<optimizer>/<model>/seed_<s>.pkl), so the figure/table
code picks them up as additional baseline methods with no changes beyond
adding their names to the BASELINES list.

Mirrors modal_app.py exactly: same four images, same task routing, the same
return-pickle-over-the-wire + save-locally pattern, resume-safe.

Usage:
    modal run modal_baselines_extra.py::extra
    modal run modal_baselines_extra.py::extra --tasks "branin"
    modal run modal_baselines_extra.py::extra --opts "tpe"
"""

import modal

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — must match modal_app.py's header so the new baselines are
#          comparable to the existing ones (same T, n_init, n_candidates...).
# ════════════════════════════════════════════════════════════════════════════
LOCAL_RESULTS_DIR = "your/own/saving/place"

# MODEL = "gpt-5.4-mini" 
# MODEL = "claude-sonnet-5"

MODEL = "claude-haiku-4-5"
T            = 61
N_INIT       = 5
N_CANDIDATES = 5
SEEDS        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

MAX_PARALLEL = 100

# The new baselines. tpe is free; the fixed_switch_* pay for T//2 LLM steps.
EXTRA_OPTS = [
    "tpe",
    "fixed_switch_llm",
    "fixed_switch_llambo",
    "fixed_switch_lgbo",
    "smac",
]

TASKS_STD = (
    ["branin", "hartmann6", "rastrigin", "schwefel", "gallagher"]
    + [f"bayesmark_{m}_{d}"
       for m in ["SVM", "RF", "DT"]
       for d in ["breast", "digits", "wine"]]
)
TASKS_YAHPO = ["yahpo_lcbench_3945", "yahpo_lcbench_7593", "yahpo_lcbench_34539"]
TASKS_LASSO = ["lasso_diabetes", "lasso_breast_cancer"]
TASKS_OLYMPUS = [
    "olympus_snar", "olympus_suzuki", "olympus_benzylation", "olympus_alkox",
    "olympus_colors_bob", "olympus_colors_n9", "olympus_fullerenes",
    "olympus_hplc", "olympus_photo_pce10", "olympus_photo_wf3",
]


# ════════════════════════════════════════════════════════════════════════════
# IMAGES — same as modal_app.py, PLUS optuna (required by the tpe baseline).
# ════════════════════════════════════════════════════════════════════════════
image_std = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy", "scipy", "scikit-learn",
        "openai", "anthropic", "tiktoken", "python-dotenv",
        "POT", "pandas", "joblib",
        # bayesmark: needed for bayesmark_* tasks. If the PyPI install pulls
        # an incompatible sklearn, pin versions here. The _sklearn_compat shim
        # in the package handles load_boston / np.float_ drift at import time.
        "bayesmark", "optuna",
        "anthropic",
    ])
    .add_local_python_source("hybrid_bo")
)


image_yahpo = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install([
        "numpy==1.26.4",
        "scipy==1.15.3",
        "pandas==2.3.3",
        "scikit-learn",
        "onnxruntime==1.23.2",
        "ConfigSpace==0.6.1",
        "yahpo_gym==1.0.2",
        "openai", "tiktoken", "python-dotenv",
        "optuna",
        "anthropic",
    ])

    .run_commands(
        "git clone --depth 1 https://github.com/slds-lmu/yahpo_data /yahpo_data"
    )
    .add_local_python_source("hybrid_bo")
)


image_lasso = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")          # celer needs a C compiler
    .run_commands("pip install --force-reinstall 'setuptools<80'")
    .pip_install([
            "setuptools<80", "wheel",
            "numpy==1.26.4", "scipy==1.15.3", "scikit-learn",
            "Cython",
            "celer", "libsvmdata", "hyperopt",
            "openai", "tiktoken", "python-dotenv", "pandas",
            "optuna",
            "anthropic",
        ])
    # sparse-ho is NOT on PyPI — install from source (LassoBench depends on it):
    .run_commands(
        "pip install git+https://github.com/QB3/sparse-ho.git"
    )
    # Install LassoBench from source (not on PyPI), server-side clone:
    .run_commands(
        "git clone https://github.com/ksehic/LassoBench.git /LassoBench "
        "&& cd /LassoBench && pip install -e ."
    )
    # Bake the datasets into the image cache so runtime needs no network:
    .run_commands(
        "python -c \""
        "from LassoBench.LassoBench import RealBenchmark; "
        "RealBenchmark(pick_data='diabetes'); "
        "RealBenchmark(pick_data='breast_cancer'); "
        "print('lasso data cached')\""
    )
    .add_local_python_source("hybrid_bo")
)

# image_olympus = (
#     modal.Image.debian_slim(python_version="3.10")
#     .pip_install([
#         "tensorflow==2.15.1", "tensorflow-probability==0.23.0",
#         "olymp",
#         "numpy==1.26.4", "scipy==1.15.3", "scikit-learn", "pandas",
#         "openai", "tiktoken", "python-dotenv",
#         "optuna",                                   # <-- TPE
#     ])
#     .add_local_python_source("hybrid_bo")
# )

image_olympus = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "matplotlib",
        "silence_tensorflow",              # ADD — olympus BNN imports it at load; its absence is the current crash
        "tensorflow==2.15.1",
        "tensorflow-probability==0.23.0",
        "tf-keras",                        # ADD — needed for the Keras 2 API the TFP layers require
        "olymp",
        "numpy==1.26.4", "scipy==1.15.3", "scikit-learn", "pandas",
        "openai", "tiktoken", "python-dotenv",
        "optuna",
        "anthropic",
    ])
    .env({"TF_USE_LEGACY_KERAS": "1"})  # ADD — forces tf.keras to Keras 2 before TF imports
    .add_local_python_source("hybrid_bo")
)

image_smac = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "swig")     # pyrfr needs these
    .pip_install([
        "smac==2.4.0",                          # pulls ConfigSpace 1.2.x
        "numpy", "scipy", "scikit-learn", "pandas",
        "openai", "tiktoken", "python-dotenv",
        "bayesmark", 
        # for bayesmark_* tasks
        "anthropic",
    ])
    .add_local_python_source("hybrid_bo")
)

image_smac_lasso = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "swig")      # swig: pyrfr (smac)
    .run_commands("pip install --force-reinstall 'setuptools<80'")
    .pip_install([
        "setuptools<80", "wheel",
        "numpy==1.26.4", "scipy==1.15.3", "scikit-learn",
        "Cython",
        "celer", "libsvmdata", "hyperopt",
        "openai", "tiktoken", "python-dotenv", "pandas",
        "smac==2.4.0",                                  # <-- added
        "anthropic",
    ])
    .run_commands("pip install git+https://github.com/QB3/sparse-ho.git")
    .run_commands(
        "git clone https://github.com/ksehic/LassoBench.git /LassoBench "
        "&& cd /LassoBench && pip install -e ."
    )
    .run_commands(
        "python -c \""
        "from LassoBench.LassoBench import RealBenchmark; "
        "RealBenchmark(pick_data='diabetes'); "
        "RealBenchmark(pick_data='breast_cancer'); "
        "print('lasso data cached')\""
    )
    .add_local_python_source("hybrid_bo")
)

image_smac_olympus = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "swig")             # swig: pyrfr (smac)
    .pip_install([
        "matplotlib",
        "silence_tensorflow",
        "tensorflow==2.15.1",
        "tensorflow-probability==0.23.0",
        "tf-keras",
        "olymp",
        "numpy==1.26.4", "scipy==1.15.3", "scikit-learn", "pandas",
        "openai", "tiktoken", "python-dotenv",
        "smac==2.4.0",                                  # <-- added
        "anthropic",
    ])
    .env({"TF_USE_LEGACY_KERAS": "1"})
    .add_local_python_source("hybrid_bo")
)
app = modal.App("hybrid-bo-baselines-extra")


# ════════════════════════════════════════════════════════════════════════════
# JOB BODY (in-container)
# ════════════════════════════════════════════════════════════════════════════
def _run_extra_baseline_job(task: str, opt: str, seed: int):
    import pickle
    from hybrid_bo.experiments.config import ExperimentConfig
    from hybrid_bo.experiments.runner import ExperimentRunner

    cfg = ExperimentConfig(
        task_name=task,
        optimizer_name=opt,
        llm_model=MODEL,
        T=T,
        n_init=N_INIT,
        n_candidates=N_CANDIDATES,
        seeds=[seed],
        tau=T,        # policy sentinel ("never switches"); the fixed_switch
                      # optimizer's OWN switch point is T//2, set in the runner.
    )
    runner = ExperimentRunner()
    result = runner.run_single(cfg, seed=seed)

    meta = dict(
        task=task, opt=opt, seed=seed,
        f_best=float(max(result.ys)),
        cost=float(sum(result.costs)),
    )
    return meta, pickle.dumps(result)


_FN_KW = dict(
    secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
    timeout=3600,
    retries=3,
    max_containers=MAX_PARALLEL,
)

@app.function(image=image_smac, **_FN_KW)
def extra_smac(task: str, opt: str, seed: int):
    import hybrid_bo._sklearn_compat
    print(f"[extra_smac] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)

@app.function(image=image_std, **_FN_KW)
def extra_std(task: str, opt: str, seed: int):
    print(f"[extra_std] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)


@app.function(image=image_yahpo, **_FN_KW)
def extra_yahpo(task: str, opt: str, seed: int):
    import hybrid_bo._configspace_compat            # shim BEFORE yahpo_gym
    from yahpo_gym import local_config
    local_config.init_config()
    local_config.set_data_path("/yahpo_data")
    print(f"[extra_yahpo] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)


@app.function(image=image_lasso, **_FN_KW)
def extra_lasso(task: str, opt: str, seed: int):
    print(f"[extra_lasso] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)


@app.function(image=image_olympus, **_FN_KW)
def extra_olympus(task: str, opt: str, seed: int):
    print(f"[extra_olympus] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)

@app.function(image=image_smac_lasso, **_FN_KW)
def extra_smac_lasso(task: str, opt: str, seed: int):
    print(f"[smac_lasso] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)


@app.function(image=image_smac_olympus, **_FN_KW)
def extra_smac_olympus(task: str, opt: str, seed: int):
    print(f"[smac_olympus] {task} {opt} seed={seed}")
    return _run_extra_baseline_job(task, opt, seed)
# ════════════════════════════════════════════════════════════════════════════
# LOCAL SIDE
# ════════════════════════════════════════════════════════════════════════════
def _local_store():
    from hybrid_bo.experiments.result_store import ResultStore
    return ResultStore(root_dir=LOCAL_RESULTS_DIR)   # results/raw/... (same as
                                                     # the existing baselines)

def _save_extra(result_bytes, task, opt, seed):
    import pickle
    from types import SimpleNamespace
    store = _local_store()
    result = pickle.loads(result_bytes)
    store_cfg = SimpleNamespace(
        task_name=task,
        optimizer_name=opt,        # -> results/raw/<task>/<opt>/<model>/seed_N.pkl
        llm_model=MODEL,
    )
    store.save(result, store_cfg, seed=seed)


def _split_tasks(task_list):
    yah   = [t for t in task_list if t.startswith("yahpo_")]
    lasso = [t for t in task_list if t.startswith("lasso_")]
    olymp = [t for t in task_list if t.startswith("olympus_")]
    std   = [t for t in task_list if not t.startswith(("yahpo_", "lasso_", "olympus_"))]
    return std, yah, lasso, olymp

@app.local_entrypoint()
def extra(tasks: str = "", opts: str = ""):
    task_list = ([t.strip() for t in tasks.split(",") if t.strip()]
                 or (TASKS_STD + TASKS_YAHPO + TASKS_LASSO + TASKS_OLYMPUS))
    opt_list = [o.strip() for o in opts.split(",") if o.strip()] or EXTRA_OPTS
    # SMAC has its own image (ConfigSpace>=1.0) and cannot run yahpo tasks,
    # so it is dispatched separately from the regular optimizers.
    run_smac = "smac" in opt_list
    opt_list_regular = [o for o in opt_list if o != "smac"]
    std_t, yahpo_t, lasso_t, olymp_t = _split_tasks(task_list)
    store = _local_store()

    # def _build(tlist):
    #     jobs = []
    #     for task in tlist:
    #         for opt in opt_list:
    #             for seed in SEEDS:
    #                 if store.exists(task, opt, MODEL, seed):
    #                     continue                       # resume-safe
    #                 jobs.append((task, opt, seed))
    #     return jobs
    
    def _build(tlist):
        jobs = []
        for task in tlist:
            for opt in opt_list_regular:          # <-- was opt_list
                for seed in SEEDS:
                    if store.exists(task, opt, MODEL, seed):
                        continue
                    jobs.append((task, opt, seed))
        return jobs

    std_jobs   = _build(std_t)
    yahpo_jobs = _build(yahpo_t)
    lasso_jobs = _build(lasso_t)
    olymp_jobs = _build(olymp_t)

    print("=" * 70)
    print("EXTRA BASELINES  (tpe + fixed_switch_{llm,llambo,lgbo})")
    print(f"  optimizers : {opt_list}")
    print(f"  seeds      : {SEEDS}   T={T}   tau(fixed_switch)={T // 2}")
    print(f"  save to    : {LOCAL_RESULTS_DIR}/raw/")
    print(f"  jobs: std={len(std_jobs)} yahpo={len(yahpo_jobs)} "
          f"lasso={len(lasso_jobs)} olympus={len(olymp_jobs)}")
    print("=" * 70)

    def _dispatch(fn, jobs, label):
        if not jobs:
            return
        done = 0
        for item in fn.starmap(jobs, return_exceptions=True):
            if isinstance(item, Exception):
                print(f"  FAILED ({label}): {item!r}")
                continue
            meta, blob = item
            _save_extra(blob, meta["task"], meta["opt"], meta["seed"])
            done += 1
            print(f"  [{label} {done}/{len(jobs)}] {meta['task']:24s} "
                  f"{meta['opt']:20s} seed={meta['seed']}  "
                  f"f_best={meta['f_best']:.4f}  cost=${meta['cost']:.5f}")

    _dispatch(extra_std,     std_jobs,   "std")
    _dispatch(extra_yahpo,   yahpo_jobs, "yahpo")
    _dispatch(extra_lasso,   lasso_jobs, "lasso")
    _dispatch(extra_olympus, olymp_jobs, "olympus")
    # ── SMAC: own image; std tasks only for now ────────────────────────────
# ── SMAC: own images per family. YAHPO is IMPOSSIBLE (ConfigSpace>=1.0
    #    for smac vs <=0.6.1 for yahpo_gym) — those tasks are skipped. ───────
    if run_smac:
        skipped_yahpo = [t for t in task_list if t.startswith("yahpo_")]
        if skipped_yahpo:
            print(f"\n[smac] SKIPPING {len(skipped_yahpo)} yahpo tasks — "
                  f"smac needs ConfigSpace>=1.0, yahpo_gym pins <=0.6.1. "
                  f"These cannot coexist in one image: {skipped_yahpo}")

        smac_std   = [t for t in task_list
                      if not t.startswith(("yahpo_", "lasso_", "olympus_"))]
        smac_lasso = [t for t in task_list if t.startswith("lasso_")]
        smac_olymp = [t for t in task_list if t.startswith("olympus_")]

        def _smac_jobs(tlist):
            return [(t, "smac", s) for t in tlist for s in SEEDS
                    if not store.exists(t, "smac", MODEL, s)]

        js, jl, jo = (_smac_jobs(smac_std), _smac_jobs(smac_lasso),
                      _smac_jobs(smac_olymp))
        print(f"\n  smac jobs: std={len(js)} lasso={len(jl)} olympus={len(jo)}")

        _dispatch(extra_smac,         js, "smac_std")
        _dispatch(extra_smac_lasso,   jl, "smac_lasso")
        _dispatch(extra_smac_olympus, jo, "smac_olympus")

    print(f"\nDONE. New baselines in {LOCAL_RESULTS_DIR}/raw/")