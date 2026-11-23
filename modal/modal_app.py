"""
Modal cloud runner for HybridBO — v2 (online Exp3 sweep + baselines).

Design (differs deliberately from the old modal_app.py):
  * NO Modal volume. Each container RUNS one job and RETURNS the pickled
    result object over the wire. The LOCAL entrypoint saves it into your
    laptop's results dir (LOCAL_RESULTS_DIR) via ResultStore. So results
    land directly in 2_HybridBO/hybrid_bo/results/raw/... exactly as if run
    locally — nothing to download, nothing to clear on the volume.
  * NO artificial throttling / startup stagger. DeepSeek flash allows 2500
    concurrent connections; one run == at most 1 in-flight request, so even
    running every job at once is well under the ceiling. MAX_PARALLEL just
    caps Modal container count for cost/debuggability.
  * Resume-safe LOCALLY: the entrypoint scans your local results dir and
    only ships jobs whose .pkl is missing.
  * Two images: standard (3.11) for branin/hartmann6/bayesmark, and a
    separate YAHPO image (3.10) — the YAHPO image block is a TODO you finish.

Usage:
    modal run modal_app.py::online
    modal run modal_app.py::baselines
    modal run modal_app.py::online   --tasks "branin,hartmann6"
    modal run modal_app.py::baselines --tasks "bayesmark_SVM_breast"
"""
import modal

# ════════════════════════════════════════════════════════════════════════════
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  HEADER CONFIG — EDIT THESE                                               │
# └──────────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════════

# WHERE results are saved on YOUR laptop (absolute path). The local entrypoint
# writes here; containers never touch it.
LOCAL_RESULTS_DIR = "place/your/own/saving/path"

MODEL   = "deepseek-v4-flash"

# MODEL = "gpt-5.4-mini"
# MODEL = "claude-haiku-4-5"
SEEDS   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        #    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        #    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        #    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        #    40, 41, 42, 43, 44, 45, 46, 47, 48, 49
           ]   # 10 seeds per (task,lambda) combo
# ARMS = ["tpe", "llm", "llambo", "lgbo"]
# ARMS = ["gp", "tpe", "llm", "llambo", "lgbo"]
# ARMS = ["gp", "lgbo"]
ARMS = ["gp", "llm", "llambo", "lgbo"]
_FULL_POOL = ["gp", "llm", "llambo", "lgbo"]
# _FULL_POOL = ["tpe", "llm", "llambo", "lgbo"]
# ── Online (Exp3) run parameters ────────────────────────────────────────────
T        = 61           # horizon
B        = 2            # regular batch width
B_FIRST  = 1            # small opening batch (ragged first batch)
ETA      = 7          # Exp3 learning rate
ALPHA    = 0.01         # uniform-mixing floor
# Dense lambda grid (0.1 .. 10) for a smooth regret-cost frontier (Fig 5):
# LAMBDAS  = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0]
# LAMBDAS = [1.5]
LAMBDAS = [10.0]

# Per-arm construction knobs (must match how baselines were built)
N_INIT       = 5
N_CANDIDATES = 5

# ── Baseline optimizers ─────────────────────────────────────────────────────
BASELINE_OPTS = ["gp", "llm", "llambo", "lgbo"]

# ── Task sets (standard image vs yahpo image) ───────────────────────────────
TASKS_STD = (
    ["branin", "hartmann6", "rastrigin", "schwefel", "gallagher"]
    + [f"bayesmark_{m}_{d}"
       for m in ["SVM", "RF", "DT"]
       for d in ["breast", "digits", "wine"]]
)
TASKS_YAHPO = [f"yahpo_lcbench_{i}" for i in ["3945", "7593", "34539"]]
TASKS_LASSO = ["lasso_diabetes", "lasso_breast_cancer"]
TASKS_OLYMPUS = [
    "olympus_snar", "olympus_suzuki", "olympus_benzylation", "olympus_alkox",
    "olympus_colors_bob", "olympus_colors_n9", "olympus_fullerenes",
    "olympus_hplc", "olympus_photo_pce10", "olympus_photo_wf3",
]
# ── Parallelism cap (flash allows 2500 concurrent; this is just a container
#    ceiling for cost/debuggability, NOT an API constraint) ─────────────────
MAX_PARALLEL = 100

# ════════════════════════════════════════════════════════════════════════════
# END HEADER CONFIG
# ════════════════════════════════════════════════════════════════════════════


# ── Standard container image (3.11): branin / hartmann6 / bayesmark ─────────
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
        "anthropic",
    ])

    .run_commands(
        "git clone --depth 1 https://github.com/slds-lmu/yahpo_data /yahpo_data"
    )
    .add_local_python_source("hybrid_bo")
)

# image_smac = (
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("build-essential", "swig")          # pyrfr needs these
#     .pip_install([
#         "smac==2.4.0",                              # pulls ConfigSpace>=1.0
#         "numpy", "scipy", "scikit-learn", "pandas",
#         "openai", "tiktoken", "python-dotenv",
#         "bayesmark",                                # for bayesmark_* tasks
#     ])
#     .add_local_python_source("hybrid_bo")
# )
app = modal.App("hybrid-bo-v2")


# ════════════════════════════════════════════════════════════════════════════
# REMOTE FUNCTIONS — each runs ONE job in a container and RETURNS the result.
# They do NOT save to disk; the local entrypoint saves to LOCAL_RESULTS_DIR.
# ════════════════════════════════════════════════════════════════════════════

def _run_online_job(task, lam, alpha, eta, seed, arms=list(ARMS)):
    """Body shared by std/yahpo online functions. Returns (meta, pickled bytes)."""
    import pickle
    from hybrid_bo.experiments.online_config import OnlineConfig
    from hybrid_bo.experiments.online_runner import OnlineExperimentRunner

    cfg = OnlineConfig(
        task_name=task, llm_model=MODEL, T=T, arms=arms,
        b=B, b_first=B_FIRST, eta=eta, lambda_=lam, alpha=alpha,
        n_init=N_INIT, n_candidates=N_CANDIDATES, seeds=[seed],
    )
    runner = OnlineExperimentRunner()
    result = runner.run_single(cfg, seed=seed)

    meta = dict(
        kind="online", task=task, lam=lam, alpha=alpha, eta=eta, seed=seed,
        f_best=float(result.f_best), cost=float(result.total_cost),
        final_arm=result.final_arm_name,
        storage_optimizer_name=cfg.storage_optimizer_name,
    )
    return meta, pickle.dumps(result)


def _run_baseline_job(task, opt, seed):
    """Body shared by std/yahpo baseline functions. Returns (meta, pickled bytes)."""
    import pickle
    from hybrid_bo.experiments.config import ExperimentConfig
    from hybrid_bo.experiments.runner import ExperimentRunner

    # Replicate run_local.py's branin special-casing for the v1 ExperimentConfig
    # (dist_to_optimum needs x_stars). Non-branin uses the default metric.
    x_stars = None
    metric = "wasserstein2"
    if task == "branin":
        from hybrid_bo.benchmarks.branin import BraninTask
        metric = "dist_to_optimum"
        x_stars = BraninTask.X_STARS

    cfg = ExperimentConfig(
        task_name=task, optimizer_name=opt, llm_model=MODEL,
        T=T, seeds=[seed], tau=T,          # tau=T => never switches (full run)
        metric=metric, n_candidates=N_CANDIDATES,
        x_stars=x_stars, aggregation="best_so_far",
    )
    runner = ExperimentRunner()
    result = runner.run_single(cfg, seed=seed)

    meta = dict(
        kind="baseline", task=task, opt=opt, seed=seed,
        f_best=float(result.f_best), cost=float(result.total_cost),
    )
    return meta, pickle.dumps(result)


# ── Standard-image remote functions ─────────────────────────────────────────
@app.function(
    image=image_std,
    secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
    timeout=1800,
    retries=3,
    max_containers=MAX_PARALLEL,
)
def online_std(task: str, lam: float, alpha: float, eta: float, seed: int):
    import hybrid_bo._sklearn_compat            # shim before any bayesmark import
    print(f"[online_std] {task} λ={lam} α={alpha} η={eta} seed={seed}")
    return _run_online_job(task, lam, alpha, eta, seed)


@app.function(
    image=image_std,
    secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
    timeout=1800,
    retries=3,
    max_containers=MAX_PARALLEL,
)
def baseline_std(task: str, opt: str, seed: int):
    import hybrid_bo._sklearn_compat
    print(f"[baseline_std] {task} {opt} seed={seed}")
    return _run_baseline_job(task, opt, seed)


# ── YAHPO-image remote functions (only usable once image_yahpo is set) ──────
if image_yahpo is not None:
    def _register_yahpo():
        """Mirror setup_yahpo.py exactly: shim first, then init + set path."""
        import hybrid_bo._configspace_compat      # MUST precede yahpo_gym import
        from yahpo_gym import local_config
        local_config.init_config()
        local_config.set_data_path("/yahpo_data")

    @app.function(
        image=image_yahpo,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def online_yahpo(task: str, lam: float, alpha: float, eta: float, seed: int):
        _register_yahpo()
        print(f"[online_yahpo] {task} λ={lam} α={alpha} η={eta} seed={seed}")
        return _run_online_job(task, lam, alpha, eta, seed)

    @app.function(
        image=image_yahpo,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def baseline_yahpo(task: str, opt: str, seed: int):
        _register_yahpo()
        print(f"[baseline_yahpo] {task} {opt} seed={seed}")
        return _run_baseline_job(task, opt, seed)

# ── LassoBench-image remote functions (only usable once image_lasso is set) ──
if image_lasso is not None:
    @app.function(
        image=image_lasso,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def online_lasso(task: str, lam: float, alpha: float, eta: float, seed: int):
        print(f"[online_lasso] {task} λ={lam} α={alpha} η={eta} seed={seed}")
        return _run_online_job(task, lam, alpha, eta, seed)

    @app.function(
        image=image_lasso,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def baseline_lasso(task: str, opt: str, seed: int):
        print(f"[baseline_lasso] {task} {opt} seed={seed}")
        return _run_baseline_job(task, opt, seed)

# ── Olympus-image remote functions (only usable once image_olympus is set) ──
if image_olympus is not None:
    @app.function(
        image=image_olympus,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def online_olympus(task: str, lam: float, alpha: float, eta: float, seed: int):
        print(f"[online_olympus] {task} λ={lam} α={alpha} η={eta} seed={seed}")
        return _run_online_job(task, lam, alpha, eta, seed)

    @app.function(
        image=image_olympus,
        secrets=[modal.Secret.from_name("hybrid-bo-secrets")],
        timeout=1800, retries=1, max_containers=MAX_PARALLEL,
    )
    def baseline_olympus(task: str, opt: str, seed: int):
        print(f"[baseline_olympus] {task} {opt} seed={seed}")
        return _run_baseline_job(task, opt, seed)
    


# ════════════════════════════════════════════════════════════════════════════
# LOCAL HELPERS (run on your laptop): resume-check + save returned results.
# ════════════════════════════════════════════════════════════════════════════

def _local_store():
    from hybrid_bo.experiments.result_store import ResultStore
    return ResultStore(root_dir=LOCAL_RESULTS_DIR)


# def _online_opt_name(lam, alpha, eta):
#     # MUST match OnlineConfig.storage_optimizer_name formatting.
#     return (f"online_hybrid"
#             f"_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}")

def _online_opt_name(lam, alpha, eta, arms=None):
    # MUST match OnlineConfig.storage_optimizer_name formatting, INCLUDING the
    # arms suffix, or resume-checks look up the wrong path and results are
    # saved over a different pool's runs.
    name = (f"online_hybrid"
            f"_lambda_{lam:.2f}_alpha_{alpha:.2f}_eta_{eta:.2f}")
    arms = list(arms) if arms is not None else list(ARMS)
    if arms != _FULL_POOL:
        name += "_arms_" + "-".join(arms)
    return name


def _save_online(result_bytes, task, lam, alpha, eta, seed):
    import pickle
    from types import SimpleNamespace
    store = _local_store()
    result = pickle.loads(result_bytes)
    store_cfg = SimpleNamespace(
        task_name=task,
        optimizer_name=_online_opt_name(lam, alpha, eta, ARMS),
        llm_model=MODEL,
    )
    store.save(result, store_cfg, seed=seed)


def _save_baseline(result_bytes, task, opt, seed):
    import pickle
    from types import SimpleNamespace
    store = _local_store()
    result = pickle.loads(result_bytes)
    store_cfg = SimpleNamespace(task_name=task, optimizer_name=opt, llm_model=MODEL)
    store.save(result, store_cfg, seed=seed)


def _split_tasks(task_list):
    """Partition into (std, yahpo, lasso, olympus)."""
    yah  = [t for t in task_list if t.startswith("yahpo_")]
    lasso = [t for t in task_list if t.startswith("lasso_")]
    olymp = [t for t in task_list if t.startswith("olympus_")]
    std  = [t for t in task_list if not t.startswith(("yahpo_", "lasso_", "olympus_"))]
    return std, yah, lasso, olymp


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def online(tasks: str = ""):
    """
    Online (Exp3) sweep over LAMBDAS × SEEDS for the given tasks.
    ALPHA/ETA fixed from header.

    Resume-safe: skips (task, λ, α, η, seed) whose .pkl already exists locally.

    Usage:
        modal run modal_app.py::online
        modal run modal_app.py::online --tasks "branin,hartmann6"
    """
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or (TASKS_STD + TASKS_YAHPO + TASKS_LASSO + TASKS_OLYMPUS)
    std_tasks, yahpo_tasks, lasso_tasks, olympus_tasks = _split_tasks(task_list)
    store = _local_store()

    # ── Build + resume-filter the STD job grid ──────────────────────────────
    std_jobs = []
    for task in std_tasks:
        for lam in LAMBDAS:
            opt_name = _online_opt_name(lam, ALPHA, ETA, ARMS)
            for seed in SEEDS:
                if store.exists(task, opt_name, MODEL, seed):
                    continue
                std_jobs.append((task, lam, ALPHA, ETA, seed))

    print(f"\n{'='*64}")
    print(f"ONLINE sweep")
    print(f"  std tasks:   {std_tasks}")
    print(f"  yahpo tasks: {yahpo_tasks}")
    print(f"  lambdas={LAMBDAS}  alpha={ALPHA}  eta={ETA}  seeds={SEEDS}")
    print(f"  T={T} b={B} b_first={B_FIRST}")
    print(f"  std jobs to run (after resume-skip): {len(std_jobs)}")
    print(f"{'='*64}")

    # ── Dispatch STD jobs in parallel; save each as it returns ──────────────
    done = 0
    for item in (online_std.starmap(std_jobs, return_exceptions=True) if std_jobs else []):
        if isinstance(item, Exception):
            print(f"  FAILED (exception): {item!r}"); continue
        meta, blob = item

        _save_online(blob, meta["task"], meta["lam"], meta["alpha"],
                     meta["eta"], meta["seed"])
        done += 1
        print(f"  [{done}/{len(std_jobs)}] saved {meta['task']} "
              f"λ={meta['lam']} seed={meta['seed']}  "
              f"f_best={meta['f_best']:.4f} cost=${meta['cost']:.5f} "
              f"final={meta['final_arm']}")

    # ── YAHPO jobs (only if the 3.10 image is finalized) ────────────────────
    if yahpo_tasks:
        if image_yahpo is None:
            print(f"\n[skip] {len(yahpo_tasks)} yahpo tasks — image_yahpo not set. "
                  f"Finish the YAHPO image block, then re-run.")
        else:
            yahpo_jobs = []
            for task in yahpo_tasks:
                for lam in LAMBDAS:
                    opt_name = _online_opt_name(lam, ALPHA, ETA, ARMS)
                    for seed in SEEDS:
                        if store.exists(task, opt_name, MODEL, seed):
                            continue
                        yahpo_jobs.append((task, lam, ALPHA, ETA, seed))
            print(f"\n  yahpo jobs to run: {len(yahpo_jobs)}")
            ydone = 0
            for item in (online_yahpo.starmap(yahpo_jobs, return_exceptions=True) if yahpo_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (yahpo): {item!r}"); continue
                meta, blob = item
                _save_online(blob, meta["task"], meta["lam"], meta["alpha"],
                             meta["eta"], meta["seed"])
                ydone += 1
                print(f"  [yahpo {ydone}/{len(yahpo_jobs)}] saved {meta['task']} "
                      f"λ={meta['lam']} seed={meta['seed']}")

    if lasso_tasks:
        if image_lasso is None:
            print(f"\n[skip] {len(lasso_tasks)} lasso tasks — image_lasso not set.")
        else:
            lasso_jobs = []
            for task in lasso_tasks:
                for lam in LAMBDAS:
                    opt_name = _online_opt_name(lam, ALPHA, ETA, ARMS)
                    for seed in SEEDS:
                        if store.exists(task, opt_name, MODEL, seed):
                            continue
                        lasso_jobs.append((task, lam, ALPHA, ETA, seed))
            print(f"\n  lasso jobs to run: {len(lasso_jobs)}")
            for item in (online_lasso.starmap(lasso_jobs, return_exceptions=True) if lasso_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (lasso): {item!r}"); continue
                meta, blob = item
                _save_online(blob, meta["task"], meta["lam"], meta["alpha"], meta["eta"], meta["seed"])

    if olympus_tasks:
        if image_olympus is None:
            print(f"\n[skip] {len(olympus_tasks)} olympus tasks — image_olympus not set.")
        else:
            olympus_jobs = []
            for task in olympus_tasks:
                for lam in LAMBDAS:
                    opt_name = _online_opt_name(lam, ALPHA, ETA, ARMS)
                    for seed in SEEDS:
                        if store.exists(task, opt_name, MODEL, seed):
                            continue
                        olympus_jobs.append((task, lam, ALPHA, ETA, seed))
            print(f"\n  olympus jobs to run: {len(olympus_jobs)}")
            for item in (online_olympus.starmap(olympus_jobs, return_exceptions=True) if olympus_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (olympus): {item!r}"); continue
                meta, blob = item
                _save_online(blob, meta["task"], meta["lam"], meta["alpha"], meta["eta"], meta["seed"])

    print(f"\nDONE. Results in {LOCAL_RESULTS_DIR}/raw/")


@app.local_entrypoint()
def baselines(tasks: str = ""):
    """
    Baseline sweep (gp/llm/llambo/lgbo) for the given tasks.
    Resume-safe: skips (task, opt, seed) already present locally.

    Usage:
        modal run modal_app.py::baselines
        modal run modal_app.py::baselines --tasks "branin,bayesmark_SVM_breast"
    """
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or (TASKS_STD + TASKS_YAHPO + TASKS_LASSO)
    std_tasks, yahpo_tasks, lasso_tasks, olympus_tasks = _split_tasks(task_list)
    store = _local_store()

    std_jobs = []
    for task in std_tasks:
        for opt in BASELINE_OPTS:
            for seed in SEEDS:
                if store.exists(task, opt, MODEL, seed):
                    continue
                std_jobs.append((task, opt, seed))

    print(f"\n{'='*64}")
    print(f"BASELINES")
    print(f"  std tasks:   {std_tasks}")
    print(f"  yahpo tasks: {yahpo_tasks}")
    print(f"  optimizers={BASELINE_OPTS}  seeds={SEEDS}  T={T}")
    print(f"  std jobs to run (after resume-skip): {len(std_jobs)}")
    print(f"{'='*64}")

    done = 0
    for item in (baseline_std.starmap(std_jobs, return_exceptions=True) if std_jobs else []):
        if isinstance(item, Exception):
            print(f"  FAILED (exception): {item!r}"); continue
        
        meta, blob = item
        _save_baseline(blob, meta["task"], meta["opt"], meta["seed"])
        done += 1
        print(f"  [{done}/{len(std_jobs)}] saved {meta['task']} {meta['opt']} "
              f"seed={meta['seed']}  f_best={meta['f_best']:.4f} "
              f"cost=${meta['cost']:.5f}")

    if yahpo_tasks:
        if image_yahpo is None:
            print(f"\n[skip] {len(yahpo_tasks)} yahpo tasks — image_yahpo not set.")
        else:
            yahpo_jobs = []
            for task in yahpo_tasks:
                for opt in BASELINE_OPTS:
                    for seed in SEEDS:
                        if store.exists(task, opt, MODEL, seed):
                            continue
                        yahpo_jobs.append((task, opt, seed))
            print(f"\n  yahpo baseline jobs to run: {len(yahpo_jobs)}")
            ydone = 0
            for item in (baseline_yahpo.starmap(yahpo_jobs, return_exceptions=True) if yahpo_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (yahpo): {item!r}"); continue
                meta, blob = item
                _save_baseline(blob, meta["task"], meta["opt"], meta["seed"])
                ydone += 1
                print(f"  [yahpo {ydone}/{len(yahpo_jobs)}] saved {meta['task']} "
                      f"{meta['opt']} seed={meta['seed']}")

    # ── LASSO baselines (only if the 3.10 lasso image is finalized) ─────────
    if lasso_tasks:
        if image_lasso is None:
            print(f"\n[skip] {len(lasso_tasks)} lasso tasks — image_lasso not set.")
        else:
            lasso_jobs = []
            for task in lasso_tasks:
                for opt in BASELINE_OPTS:
                    for seed in SEEDS:
                        if store.exists(task, opt, MODEL, seed):
                            continue
                        lasso_jobs.append((task, opt, seed))
            print(f"\n  lasso baseline jobs to run: {len(lasso_jobs)}")
            ldone = 0
            for item in (baseline_lasso.starmap(lasso_jobs, return_exceptions=True) if lasso_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (lasso): {item!r}"); continue
                meta, blob = item
                _save_baseline(blob, meta["task"], meta["opt"], meta["seed"])
                ldone += 1
                print(f"  [lasso {ldone}/{len(lasso_jobs)}] saved {meta['task']} "
                      f"{meta['opt']} seed={meta['seed']}")
    if olympus_tasks:
        if image_olympus is None:
            print(f"\n[skip] {len(olympus_tasks)} olympus tasks — image_olympus not set.")
        else:
            olympus_jobs = []
            for task in olympus_tasks:
                for opt in BASELINE_OPTS:
                    for seed in SEEDS:
                        if store.exists(task, opt, MODEL, seed):
                            continue
                        olympus_jobs.append((task, opt, seed))
            print(f"\n  olympus baseline jobs to run: {len(olympus_jobs)}")
            for item in (baseline_olympus.starmap(olympus_jobs, return_exceptions=True) if olympus_jobs else []):
                if isinstance(item, Exception):
                    print(f"  FAILED (olympus): {item!r}"); continue
                meta, blob = item
                _save_baseline(blob, meta["task"], meta["opt"], meta["seed"])

    print(f"\nDONE. Results in {LOCAL_RESULTS_DIR}/raw/")
    