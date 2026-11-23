from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


_VALID_YAHPO_SCENARIOS  = {"lcbench"}
_VALID_YAHPO_INSTANCES  = {
    "3945", "7593", "34539", "126025", "126026",
    "146212", "167190", "167201", "168330", "168868",
}

@dataclass
class ExperimentConfig:
    """
    Single source of truth for one experiment run.

    Fields
    ------
    task_name       : "branin" | "hartmann6" |
                      "bayesmark_{model}_{dataset}"
                      e.g. "bayesmark_SVM_breast", "bayesmark_RF_digits"
    optimizer_name  : "gp" | "llm" | "llambo" | "lgbo" |
                      "hybrid_llm" | "hybrid_llambo" | "hybrid_lgbo"
    llm_model       : "gpt-3.5-turbo" | "gpt-4-turbo" | "gpt-4o" |
                      "claude-3-5-sonnet" | "deepseek-chat" | "llama-local"
    T               : total BO iterations
    n_init          : warm-up steps before GP can fit
    seeds           : list of random seeds for repeated runs
    epsilon         : switching proportion ε ∈ (0,1) — use EpsilonPolicy
    tau             : fixed switching step — use FixedTauPolicy
                      exactly one of epsilon / tau must be set
    metric          : distance metric for EpsilonPolicy
    window          : trajectory window size for distance computation
    min_steps       : minimum steps before EpsilonPolicy can fire
    n_candidates    : candidate pool size for LLAMBOOptimizer
    x_stars         : shape (k, dim) optimum locations for dist_to_optimum
                      required when metric="dist_to_optimum"
    aggregation     : aggregation for dist_to_optimum metric
    """

    task_name:      str
    optimizer_name: str
    llm_model:      str                 = "gpt-3.5-turbo"
    T:              int                 = 30
    n_init:         int                 = 5
    seeds:          List[int]           = field(default_factory=lambda: [0, 1, 2])
    epsilon:        Optional[float]     = None
    tau:            Optional[int]       = None
    metric:         str                 = "wasserstein2"
    window:         int                 = 10
    min_steps:      int                 = 5
    n_candidates:   int                 = 20
    x_stars:        Optional[np.ndarray] = None
    aggregation:    str                 = "best_so_far"

    # ── Valid value sets ───────────────────────────────────────────────────────

    _VALID_SYNTHETIC_TASKS = {"branin", "hartmann6", "rastrigin", "schwefel", "gallagher"}

    _VALID_BAYESMARK_MODELS = {
        "SVM", "RF", "DT", "MLP-adam", "MLP-sgd", "ada", "kNN"
    }

    _VALID_BAYESMARK_DATASETS = {
        "breast", "digits", "iris", "wine", "diabetes"
    }

    _VALID_OPTIMIZERS = {
        "gp", "llm", "llambo", "lgbo",
        "hybrid_llm", "hybrid_llambo", "hybrid_lgbo",
        "tpe",
        "fixed_switch_llm", "fixed_switch_llambo", "fixed_switch_lgbo",
        "smac",
    }

    _VALID_METRICS = {
        "wasserstein2", "mmd", "kl", "energy",
        "pointwise", "dist_to_optimum",
    }

    _VALID_AGGREGATIONS = {"best_so_far", "mean", "last"}

    _VALID_YAHPO_SCENARIOS  = {"lcbench"}
    _VALID_YAHPO_INSTANCES  = {
        "3945", "7593", "34539", "126025", "126026",
        "146212", "167190", "167201", "168330", "168868",
    }
    # ── Validation ─────────────────────────────────────────────────────────────

    def __post_init__(self):
        self._validate_task()
        self._validate_optimizer()
        self._validate_metric()
        self._validate_policy()

    def _validate_task(self):
        if self.task_name in self._VALID_SYNTHETIC_TASKS:
            return   # branin, hartmann6 — always valid
        
        if self.task_name.startswith(("rastrigin", "schwefel", "gallagher")):
            parts = self.task_name.split("_")
            if len(parts) > 1:
                try:
                    int(parts[1])
                except ValueError:
                    raise ValueError(
                        f"Multimodal task dim suffix must be an integer, "
                        f"got '{self.task_name}'.")
                    
        if self.task_name.startswith("bayesmark_"):
            parts = self.task_name.split("_", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"Bayesmark task name must be 'bayesmark_{{model}}_{{dataset}}', "
                    f"got '{self.task_name}'."
                )
            _, model, dataset = parts
            if model not in self._VALID_BAYESMARK_MODELS:
                raise ValueError(
                    f"Unknown Bayesmark model '{model}'. "
                    f"Choose from {sorted(self._VALID_BAYESMARK_MODELS)}."
                )
            if dataset not in self._VALID_BAYESMARK_DATASETS:
                raise ValueError(
                    f"Unknown Bayesmark dataset '{dataset}'. "
                    f"Choose from {sorted(self._VALID_BAYESMARK_DATASETS)}."
                )
            return   # valid bayesmark task
        
        if self.task_name.startswith("yahpo_"):
            parts = self.task_name.split("_", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"YAHPO task must be 'yahpo_{{scenario}}_{{instance}}', "
                    f"got '{self.task_name}'."
                )
            _, scenario, instance = parts
            if scenario not in self._VALID_YAHPO_SCENARIOS:
                raise ValueError(
                    f"Unknown YAHPO scenario '{scenario}'. "
                    f"Choose from {sorted(self._VALID_YAHPO_SCENARIOS)}."
                )
            if instance not in self._VALID_YAHPO_INSTANCES:
                raise ValueError(
                    f"Unknown YAHPO instance '{instance}'. "
                    f"Choose from {sorted(self._VALID_YAHPO_INSTANCES)}."
                )
            return
        
        if self.task_name in ("lasso_diabetes", "lasso_breast_cancer"):
            return
        
        if self.task_name.startswith("olympus_"):
            return
        
        raise ValueError(
            f"Unknown task '{self.task_name}'. "
            f"Use one of {sorted(self._VALID_SYNTHETIC_TASKS)} "
            f"or 'bayesmark_{{model}}_{{dataset}}'."
        )

    def _validate_optimizer(self):
        if self.optimizer_name not in self._VALID_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer_name}'. "
                f"Choose from {sorted(self._VALID_OPTIMIZERS)}."
            )

    def _validate_metric(self):
        if self.metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{self.metric}'. "
                f"Choose from {sorted(self._VALID_METRICS)}."
            )
        if self.metric == "dist_to_optimum" and self.x_stars is None:
            raise ValueError(
                "metric='dist_to_optimum' requires x_stars to be set. "
                "Pass the array of global optimum locations."
            )
        if self.aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{self.aggregation}'. "
                f"Choose from {sorted(self._VALID_AGGREGATIONS)}."
            )

    def _validate_policy(self):
        if self.epsilon is None and self.tau is None:
            raise ValueError(
                "Provide either epsilon (ProportionalEpsilonPolicy) "
                "or tau (FixedTauPolicy) — not both None."
            )
        if self.epsilon is not None and self.tau is not None:
            raise ValueError(
                "Provide either epsilon or tau — not both. "
                f"Got epsilon={self.epsilon}, tau={self.tau}."
            )
        if self.epsilon is not None and not (0 < self.epsilon < 1):
            raise ValueError(
                f"epsilon must be a proportion in (0, 1). Got {self.epsilon}."
            )
        if self.epsilon is None and self.tau is None:
            raise ValueError("Provide either epsilon ... or tau ... — not both None.")

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def is_bayesmark(self) -> bool:
        return self.task_name.startswith("bayesmark_")

    @property
    def is_synthetic(self) -> bool:
            return (self.task_name in self._VALID_SYNTHETIC_TASKS
                    or self.task_name.startswith(("rastrigin", "schwefel", "gallagher")))
    # def is_synthetic(self) -> bool:
    #     return self.task_name in self._VALID_SYNTHETIC_TASKS

    @property
    def bayesmark_model(self) -> Optional[str]:
        """e.g. 'SVM' from 'bayesmark_SVM_breast'. None if not bayesmark."""
        if not self.is_bayesmark:
            return None
        return self.task_name.split("_", 2)[1]

    @property
    def bayesmark_dataset(self) -> Optional[str]:
        """e.g. 'breast' from 'bayesmark_SVM_breast'. None if not bayesmark."""
        if not self.is_bayesmark:
            return None
        return self.task_name.split("_", 2)[2]

    @property
    def uses_epsilon_policy(self) -> bool:
        return self.epsilon is not None

    @property
    def uses_fixed_tau(self) -> bool:
        return self.tau is not None
    
    @property
    def is_yahpo(self) -> bool:
        return self.task_name.startswith("yahpo_")

    @property
    def yahpo_scenario(self) -> Optional[str]:
        if not self.is_yahpo:
            return None
        return self.task_name.split("_", 2)[1]

    @property
    def yahpo_instance(self) -> Optional[str]:
        if not self.is_yahpo:
            return None
        return self.task_name.split("_", 2)[2]