"""
YAHPOTask: wraps YAHPO Gym's lcbench scenario as a BaseTask.

Each instance (OpenML dataset ID) is one task.
Search space: lcbench hyperparameters, normalized to [0,1]^d.
Objective: val_accuracy (maximise).

One-time setup required:
    git clone https://github.com/slds-lmu/yahpo_data ~/yahpo_data
    python -c "
        from yahpo_gym import local_config
        local_config.init_config()
        local_config.set_data_path('~/yahpo_data')
    "
"""
import numpy as np
from typing import List, Tuple, Optional
from .base_task import BaseTask


LCBENCH_SPACE = {
    "batch_size":       (16,    512,   "log_int"),
    "learning_rate":    (1e-4,  1e-1,  "log_float"),
    "momentum":         (0.1,   0.99,  "float"),
    "weight_decay":     (1e-5,  1e-1,  "log_float"),
    "num_layers":       (1,     5,     "int"),
    "max_units":        (64,    512,   "log_int"),
    "max_dropout":      (0.0,   1.0,   "float"),
}

# Recommended instances from LLAMBO paper
LCBENCH_INSTANCES = [
    "3945", "7593", "34539", "126025", "126026",
    "146212", "167190", "167201", "168330", "168868",
]


class YAHPOTask(BaseTask):
    """
    YAHPO lcbench task for one OpenML dataset instance.

    Args:
        instance:    OpenML dataset ID string, e.g. "3945"
        scenario:    YAHPO scenario name (default: "lcbench")
        objective:   which YAHPO objective to maximise (default: "val_accuracy")
        epoch:       fidelity — which training epoch to query (default: 52 = max)
    """

    def __init__(
        self,
        instance:  str   = "3945",
        scenario:  str   = "lcbench",
        objective: str   = "val_accuracy",
        epoch:     int   = 52,
    ):
        from yahpo_gym import benchmark_set

        self.instance    = instance
        self.scenario    = scenario
        self.objective   = objective
        self.epoch       = epoch
        self.name        = f"yahpo_{scenario}_{instance}"

        # Initialise YAHPO benchmark
        self._bench = benchmark_set.BenchmarkSet(scenario)
        self._bench.set_instance(instance)

        # Build our search space from LCBENCH_SPACE
        self._param_names = list(LCBENCH_SPACE.keys())
        self._space       = LCBENCH_SPACE
        self.dim          = len(self._param_names)
        self.bounds       = [(0.0, 1.0)] * self.dim   # normalised to [0,1]^d

        # Cache best known value (approximated from surrogate grid)
        self._f_star_cache: Optional[float] = None

    # ── Core interface ─────────────────────────────────────────────────────────

    def _unscale(self, x: np.ndarray) -> dict:
        """Convert x from [0,1]^d to the real hyperparameter dict."""
        params = {}
        for i, name in enumerate(self._param_names):
            lo, hi, scale = self._space[name]
            v = float(np.clip(x[i], 0.0, 1.0))

            if "log" in scale:
                real = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
            else:
                real = lo + v * (hi - lo)

            if "int" in scale:
                real = int(round(real))
            else:
                # guard against float rounding pushing values a hair outside [lo, hi]
                real = float(np.clip(real, lo, hi))

            params[name] = real
        
        params["OpenML_task_id"] = str(self.instance)
        # YAHPO lcbench also needs epoch as fidelity
        params["epoch"] = int(self.epoch)
        return params

    def evaluate(self, x: np.ndarray) -> float:
        """
        Query YAHPO surrogate. Returns objective value (higher = better).
        Extremely fast (~1ms) since it's a surrogate neural net.
        """
        params = self._unscale(x)
        try:
            result = self._bench.objective_function(params)
            # result is a list of dicts; take first
            val = result[0][self.objective]
            return float(val)
        except Exception as e:
            print(f"[YAHPOTask] evaluate failed: {e}  params={params}")
            raise

    def f_star(self) -> float:
        """
        Approximate f* by evaluating on a random grid of 2000 points.
        Cached after first call — fast because YAHPO surrogate is instant.
        """
        if self._f_star_cache is not None:
            return self._f_star_cache

        rng  = np.random.default_rng(0)
        xs   = rng.uniform(0, 1, size=(2000, self.dim))
        ys   = [self.evaluate(xs[i]) for i in range(len(xs))]
        self._f_star_cache = float(max(ys))
        return self._f_star_cache

    def random_x(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=self.dim)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def param_description(self) -> str:
        """Rich description for LLM prompts — includes semantic meaning."""
        lines = [
            f"Hyperparameter optimization for a neural network on "
            f"OpenML dataset {self.instance} (lcbench benchmark).",
            f"Objective: maximize {self.objective}.",
            f"Search space ({self.dim} parameters, all normalized to [0,1]):",
        ]
        descriptions = {
            "batch_size":    "batch size for SGD training (log scale, 16–512)",
            "learning_rate": "SGD learning rate (log scale, 1e-4 to 1e-1)",
            "momentum":      "SGD momentum (0.1–0.99)",
            "weight_decay":  "L2 regularization weight decay (log scale, 1e-5 to 1e-1)",
            "num_layers":    "number of hidden layers (1–5, integer)",
            "max_units":     "max units per hidden layer (log scale, 64–512, integer)",
            "max_dropout":   "maximum dropout rate (0.0–1.0)",
        }
        for name in self._param_names:
            lines.append(f"  {name}: {descriptions.get(name, '')}")
        return "\n".join(lines)