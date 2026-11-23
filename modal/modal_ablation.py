"""
modal_ablation.py — run the ABLATION variants on Modal (one-time-switch removed).

Mirrors modal_app.py exactly (same images, same job body, same
return-pickle-over-the-wire + save-locally pattern), but:

  * runs the ABLATED variants, NOT "Full HybridBO" (Full is your main
    sweep, already in results/raw/ — it is not re-run here);
  * saves everything under  <results>/ablation/  so ablation output never
    mixes with the main results;
  * routes each task to the correct image (std / yahpo / lasso / olympus),
    exactly like modal_app.py.

Variants (each = ONE change from Full):
    no_cost              lambda_ = 0.0        (cost coefficient removed)
    no_exploration       alpha   = 0.0        (uniform exploration removed)
    fixed_distribution   eta     = 0.0        (weights never update -> P stays
                                               uniform == "random switching")
    no_gp                arms = [llm, llambo, lgbo]
    no_llm               arms = [gp, llambo, lgbo]
    no_llambo            arms = [gp, llm, lgbo]
    no_lgbo              arms = [gp, llm, llambo]

"fixed_distribution" and "random switching" are the SAME mechanism (eta=0 =>
weights frozen => P stays uniform => uniform sampling). One run; report two
table rows from it.

Usage:
    modal run modal_ablation.py::ablation
    modal run modal_ablation.py::ablation --tasks "branin"
    modal run modal_ablation.py::ablation --variants "no_gp,no_cost"
"""

import modal

# ════════════════════════════════════════════════════════════════════════════
# CONFIG  — keep BASE identical to the main sweep, or the ablation is not a
#          one-change-at-a-time comparison.
# ════════════════════════════════════════════════════════════════════════════
LOCAL_RESULTS_DIR = "place/your/own/saving/place"

MODEL   = "deepseek-v4-flash"

T        = 61
B        = 2
B_FIRST  = 1
ETA      = 7            # Full's eta   (must match modal_app.py)
ALPHA    = 0.01         # Full's alpha (must match modal_app.py)

N_INIT       = 5
N_CANDIDATES = 5

FULL_POOL = ["gp", "llm", "llambo", "lgbo"]

TASKS_STD = (
    ["branin", "hartmann6", "rastrigin", "schwefel", "gallagher"]
    + [f"bayesmark_{m}_{d}"
       for m in ["SVM", "RF", "DT"]
       for d in ["breast", "digits", "wine"]]
)
TASKS_YAHPO   = [f"yahpo_lcbench_{i}" for i in ["3945", "7593", "34539"]]
TASKS_LASSO   = ["lasso_diabetes", "lasso_breast_cancer"]
TASKS_OLYMPUS = [
    "olympus_snar", "olympus_suzuki", "olympus_benzylation", "olympus_alkox",
    "olympus_colors_bob", "olympus_colors_n9", "olympus_fullerenes",
    "olympus_hplc", "olympus_photo_pce10", "olympus_photo_wf3",
]

# ABL_TASKS = TASKS_STD + TASKS_YAHPO + TASKS_LASSO + TASKS_OLYMPUS
ABL_TASKS = ["hartmann6", "schwefel","bayesmark_SVM_breast","bayesmark_SVM_digits","lasso_diabetes","lasso_breast_cancer","yahpo_lcbench_7593","olympus_suzuki"]
ABL_LAMBDAS = [0.1, 1.0, 10.0]
ABL_SEEDS   = [0, 1, 2, 3, 4]

ALL_VARIANTS = [
    "no_exploration",
    "fixed_distribution",     # == random switching (same mechanism)
    "no_gp",
    "no_llm",
    "no_llambo",
    "no_lgbo",
]

# no_cost IS lambda=0 by definition, so it is NOT swept over lambda.
LAMBDA_FIXED_VARIANTS = {"no_cost"}

MAX_PARALLEL = 100


# ════════════════════════════════════════════════════════════════════════════
# VARIANT -> config overrides
# ════════════════════════════════════════════════════════════════════════════
def _variant_overrides(variant: str) -> dict:
    if variant == "no_cost":
        return dict(lambda_=0.0)
    if variant == "no_exploration":
        return dict(alpha=0.0)
    if variant == "fixed_distribution":
        return dict(eta=0.0)
    if variant == "no_gp":
        return dict(arms=["llm", "llambo", "lgbo"])
    if variant == "no_llm":
        return dict(arms=["gp", "llambo", "lgbo"])
    if variant == "no_llambo":
        return dict(arms=["gp", "llm", "lgbo"])
    if variant == "no_lgbo":
        return dict(arms=["gp", "llm", "llambo"])
    raise ValueError(f"Unknown variant '{variant}'.")


def _variant_storage_name(variant: str, lam: float) -> str:
    """Reconstruct OnlineConfig.storage_optimizer_name LOCALLY (for resume
    checks and for saving), without importing hybrid_bo on the client.
    MUST stay in lockstep with OnlineConfig.storage_optimizer_name.
    """
    o = _variant_overrides(variant)
    eff_lam   = o.get("lambda_", lam)
    eff_alpha = o.get("alpha", ALPHA)
    eff_eta   = o.get("eta", ETA)
    base = (f"online_hybrid"
            f"_lambda_{eff_lam:.2f}"
            f"_alpha_{eff_alpha:.2f}"
            f"_eta_{eff_eta:.2f}")

    arms = o.get("arms", FULL_POOL)
    if list(arms) != FULL_POOL:
        base += "_arms_" + "-".join(arms)
    return base


# ════════════════════════════════════════════════════════════════════════════
# IMAGES — identical to modal_app.py (copy the definitions verbatim).
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
    ])
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
    ])
    .env({"TF_USE_LEGACY_KERAS": "1"})  # ADD — forces tf.keras to Keras 2 before TF imports
    .add_local_python_source("hybrid_bo")
)
# ── YAHPO container image (3.10): TODO — you finish this block ───────────────
# YAHPO needs Python 3.10, yahpo_gym, a specific ConfigSpace version (the
# _configspace_compat shim covers drift), AND the yahpo_data repo present in
# the image. On your laptop this lives in a separate .venv-yahpo; reproduce it
# here. Sketch (fill in / verify versions):

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
    ])
    .run_commands(
        "git clone --depth 1 https://github.com/slds-lmu/yahpo_data /yahpo_data"
    )
    .add_local_python_source("hybrid_bo")
)
app = modal.App("hybrid-bo-ablation")


# ════════════════════════════════════════════════════════════════════════════
# JOB BODY (runs inside the container)
# ════════════════════════════════════════════════════════════════════════════
def _run_ablation_job(task: str, variant: str, lam: float, seed: int):
    """Build the variant's OnlineConfig and run it. Returns (meta, pickle)."""
    import pickle
    from hybrid_bo.experiments.online_config import OnlineConfig
    from hybrid_bo.experiments.online_runner import OnlineExperimentRunner

    kwargs = dict(
        task_name=task, llm_model=MODEL, T=T,
        b=B, b_first=B_FIRST,
        eta=ETA, lambda_=lam, alpha=ALPHA,
        arms=list(FULL_POOL),
        n_init=N_INIT, n_candidates=N_CANDIDATES,
        seeds=[seed],
    )
    kwargs.update(_variant_overrides(variant))

    cfg = OnlineConfig(**kwargs)
    runner = OnlineExperimentRunner()
    result = runner.run_single(cfg, seed=seed)

    meta = dict(
        kind="ablation", task=task, variant=variant, lam=lam, seed=seed,
        f_best=float(result.f_best), cost=float(result.total_cost),
        final_arm=result.final_arm_name,
        storage_optimizer_name=cfg.storage_optimizer_name,
    )
    return meta, pickle.dumps(result)


# ════════════════════════════════════════════════════════════════════════════
# REMOTE FUNCTIONS — one per image family
# ════════════════════════════════════════════════════════════════════════════
_FN_KW = dict(
    secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
    timeout=3600,
    retries=3,
    max_containers=MAX_PARALLEL,
)


@app.function(image=image_std, **_FN_KW)
def ablation_std(task: str, variant: str, lam: float, seed: int):
    print(f"[abl_std] {task} {variant} lam={lam} seed={seed}")
    return _run_ablation_job(task, variant, lam, seed)


@app.function(image=image_yahpo, **_FN_KW)
def ablation_yahpo(task: str, variant: str, lam: float, seed: int):
    import hybrid_bo._configspace_compat            # shim BEFORE yahpo_gym
    from yahpo_gym import local_config
    local_config.init_config()
    local_config.set_data_path("/yahpo_data")
    print(f"[abl_yahpo] {task} {variant} lam={lam} seed={seed}")
    return _run_ablation_job(task, variant, lam, seed)


@app.function(image=image_lasso, **_FN_KW)
def ablation_lasso(task: str, variant: str, lam: float, seed: int):
    print(f"[abl_lasso] {task} {variant} lam={lam} seed={seed}")
    return _run_ablation_job(task, variant, lam, seed)


@app.function(image=image_olympus, **_FN_KW)
def ablation_olympus(task: str, variant: str, lam: float, seed: int):
    print(f"[abl_olympus] {task} {variant} lam={lam} seed={seed}")
    return _run_ablation_job(task, variant, lam, seed)


# ════════════════════════════════════════════════════════════════════════════
# LOCAL-SIDE HELPERS (save + routing)
# ════════════════════════════════════════════════════════════════════════════
def _local_store():
    from hybrid_bo.experiments.result_store import ResultStore
    return ResultStore(root_dir=LOCAL_RESULTS_DIR)     # -> results/ablation/raw/...


def _save_ablation(result_bytes, task, variant, lam, seed):
    import pickle
    from types import SimpleNamespace
    store = _local_store()
    result = pickle.loads(result_bytes)
    store_cfg = SimpleNamespace(
        task_name=task,
        optimizer_name=_variant_storage_name(variant, lam),
        llm_model=MODEL,
    )
    store.save(result, store_cfg, seed=seed)           # (result, cfg, seed)


def _split_tasks(task_list):
    """Partition into (std, yahpo, lasso, olympus) — same rule as modal_app."""
    yah   = [t for t in task_list if t.startswith("yahpo_")]
    lasso = [t for t in task_list if t.startswith("lasso_")]
    olymp = [t for t in task_list if t.startswith("olympus_")]
    std   = [t for t in task_list if not t.startswith(("yahpo_", "lasso_", "olympus_"))]
    return std, yah, lasso, olymp


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════
@app.local_entrypoint()
def ablation(tasks: str = "", variants: str = "", lambdas: str = ""):
    task_list    = [t.strip() for t in tasks.split(",") if t.strip()] or ABL_TASKS
    variant_list = [v.strip() for v in variants.split(",") if v.strip()] or ALL_VARIANTS
    lambda_list  = [float(x) for x in lambdas.split(",") if x.strip()] or ABL_LAMBDAS

    std_t, yahpo_t, lasso_t, olymp_t = _split_tasks(task_list)
    store = _local_store()

    def _build_jobs(tlist):
        jobs = []
        for task in tlist:
            for variant in variant_list:
                lams = [0.0] if variant in LAMBDA_FIXED_VARIANTS else lambda_list
                for lam in lams:
                    opt_name = _variant_storage_name(variant, lam)
                    for seed in ABL_SEEDS:
                        if store.exists(task, opt_name, MODEL, seed):
                            continue                       # resume-safe
                        jobs.append((task, variant, lam, seed))
        return jobs

    std_jobs   = _build_jobs(std_t)
    yahpo_jobs = _build_jobs(yahpo_t)
    lasso_jobs = _build_jobs(lasso_t)
    olymp_jobs = _build_jobs(olymp_t)

    print("=" * 70)
    print("ABLATION (Full HybridBO is NOT re-run — it is your main sweep)")
    print(f"  tasks    : std={std_t} yahpo={yahpo_t} lasso={lasso_t} olympus={olymp_t}")
    print(f"  variants : {variant_list}")
    print(f"  lambdas  : {lambda_list}   (no_cost fixed at lambda=0)")
    print(f"  seeds    : {ABL_SEEDS}")
    print(f"  base     : T={T} b={B} b_first={B_FIRST} alpha={ALPHA} eta={ETA}")
    print(f"  save to  : {LOCAL_RESULTS_DIR}/raw/")
    print(f"  jobs (after resume-skip): std={len(std_jobs)} yahpo={len(yahpo_jobs)} "
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
            _save_ablation(blob, meta["task"], meta["variant"],
                           meta["lam"], meta["seed"])
            done += 1
            print(f"  [{label} {done}/{len(jobs)}] {meta['task']:22s} "
                  f"{meta['variant']:20s} lam={meta['lam']:<5} seed={meta['seed']} "
                  f"f_best={meta['f_best']:.4f} cost=${meta['cost']:.5f}")

    _dispatch(ablation_std,     std_jobs,   "std")
    _dispatch(ablation_yahpo,   yahpo_jobs, "yahpo")
    _dispatch(ablation_lasso,   lasso_jobs, "lasso")
    _dispatch(ablation_olympus, olymp_jobs, "olympus")

    print(f"\nDONE. Ablation results in {LOCAL_RESULTS_DIR}/raw/")