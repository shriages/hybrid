"""
BayesmarkTask: wraps Bayesmark's SklearnModel as a BaseTask.

Each (model, dataset) pair is one task. The search space is the
hyperparameter space defined by Bayesmark for that model.

Supported models:   SVM, RF, DT, MLP-adam, MLP-sgd, ada, kNN
Supported datasets: breast, digits, iris, wine, diabetes
                    (classification: breast, digits, iris, wine)
                    (regression:     diabetes)
"""
from hybrid_bo import _sklearn_compat
import numpy as np
from typing import List, Tuple
from .base_task import BaseTask

SEARCH_SPACES = {
    "SVM": {
        "C":       (1.0,    1e3,  "log"),
        "gamma":   (1e-4,   1e-3, "log"),
        "tol":     (1e-5,   1e-1, "log"),
    },
    "RF": {
        "max_depth":        (1,    15,   "linear"),
        "max_features":     (0.01, 0.99, "linear"),
        "min_samples_split":(2,    20,   "linear"),
        "min_samples_leaf": (1,    20,   "linear"),
        "n_estimators":     (10,   100,  "linear"),
        "bootstrap":        (0,    1,    "linear"),   # 0=False, 1=True
    },
    "DT": {
        "max_depth":        (1,    15,   "linear"),
        "max_features":     (0.01, 0.99, "linear"),
        "min_samples_split":(2,    20,   "linear"),
        "min_samples_leaf": (1,    20,   "linear"),
        "min_impurity_decrease": (0.0, 0.5, "linear"),
        "criterion":        (0,    1,    "linear"),  # 0=gini, 1=entropy
    },
    "MLP-adam": {
        "hidden_layer_sizes": (50,  200,  "linear"),
        "alpha":              (1e-5, 1e-1, "log"),
        "learning_rate_init": (1e-4, 1e-1, "log"),
        "tol":                (1e-5, 1e-1, "log"),
        "max_iter":           (50,   200,  "linear"),
        "beta_1":             (0.8,  0.99, "linear"),
        "beta_2":             (0.8,  0.999,"linear"),
    },
    "ada": {
        "n_estimators":  (10,  100,  "linear"),
        "learning_rate": (1e-4, 1e0, "log"),
    },
}

# Which metric to maximise per dataset type
# Bayesmark returns accuracy for clf, negative MSE for reg (so higher = better)
DATASET_METRICS = {
    "breast":   "acc",
    "digits":   "acc",
    "iris":     "acc",
    "wine":     "acc",
    "diabetes": "mse",    # we negate MSE so higher = better
}


class BayesmarkTask(BaseTask):
    """
    One (model, dataset) HPO task from Bayesmark.

    evaluate(x) runs sklearn cross-validation and returns CV score.
    Higher is always better (accuracy for clf, negated MSE for reg).

    Args:
        model_name:  one of SEARCH_SPACES keys ("SVM", "RF", "DT", etc.)
        dataset:     one of DATASET_METRICS keys ("breast", "digits", etc.)
        shuffle_seed: sklearn CV split seed (fixed for reproducibility)
    """

    def __init__(
        self,
        model_name:   str = "SVM",
        dataset:      str = "breast",
        shuffle_seed: int = 0,
    ):
        from bayesmark.sklearn_funcs import SklearnModel

        assert model_name in SEARCH_SPACES, \
            f"Unknown model '{model_name}'. Choose from {list(SEARCH_SPACES)}"
        assert dataset in DATASET_METRICS, \
            f"Unknown dataset '{dataset}'. Choose from {list(DATASET_METRICS)}"

        self.model_name   = model_name
        self.dataset      = dataset
        self.shuffle_seed = shuffle_seed
        self.name         = f"bayesmark_{model_name}_{dataset}"

        # Metric: acc for classification, mse for regression
        metric = DATASET_METRICS[dataset]
        self._is_regression = (metric == "mse")
        bm_metric = "mse" if self._is_regression else "acc"

        # Bayesmark model wrapper
        self._bm_model = SklearnModel(
            model      = model_name,
            dataset    = dataset,
            metric     = bm_metric,
            shuffle_seed = shuffle_seed,
        )
        _original_scorer = self._bm_model.scorer
        def _scorer_compat(*args, **kwargs):
            return np.float64(_original_scorer(*args, **kwargs))
        self._bm_model.scorer = _scorer_compat
        
        # Build bounds from search space
        self._param_names = list(SEARCH_SPACES[model_name].keys())
        self._space       = SEARCH_SPACES[model_name]
        self.dim          = len(self._param_names)
        self.bounds       = [(0.0, 1.0)] * self.dim   # always [0,1]^d

    def _unscale(self, x: np.ndarray) -> dict:
        """
        Convert x from [0,1]^d to the real hyperparameter dict
        that sklearn expects.
        """
        params = {}
        for i, name in enumerate(self._param_names):
            lo, hi, scale = self._space[name]
            v = float(x[i])
            v = np.clip(v, 0.0, 1.0)

            if scale == "log":
                # Map [0,1] → [log(lo), log(hi)] → exponentiate
                real = np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))
            else:
                real = lo + v * (hi - lo)

            # Integer parameters
            if name in ("max_depth", "min_samples_split", "min_samples_leaf",
                        "n_estimators", "hidden_layer_sizes", "max_iter"):
                real = int(round(real))

            # Boolean parameters
            if name == "bootstrap":
                real = bool(round(real))

            # Categorical: criterion
            if name == "criterion":
                real = "gini" if v < 0.5 else "entropy"

            params[name] = real
        return params

    def evaluate(self, x: np.ndarray) -> float:
        """
        Run sklearn CV with hyperparameters x. Returns scalar (higher = better).
        For regression (MSE), returns negative MSE so higher is still better.
        """
        params = self._unscale(x)
        try:
            # Bayesmark evaluate returns array of shape (1,) or scalar
            result = self._bm_model.evaluate(params)
            score  = float(np.atleast_1d(result)[0])  # return cv_loss=-cv_score, so it returns -xxx
            # For MSE: negate so higher = better
            if not self._is_regression:
                score = -score
            return score
        except Exception as e:
            # Return a very bad score on failure (invalid HP combination)
            print(f"[BayesmarkTask] evaluate failed: {e} params={params}")
            return -1e6 if not self._is_regression else -1e6

    def f_star(self) -> float:
        """
        True optimum unknown for real tasks.
        Return None — regret will be reported as gap from best found,
        not from true optimum.
        """
        return None

    def random_x(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform random in [0,1]^d."""
        return rng.uniform(0.0, 1.0, size=self.dim)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)

    def param_description(self) -> str:
        """Natural language description for LLM prompts."""
        lines = [f"Hyperparameter optimization for {self.model_name} "
                 f"on the {self.dataset} dataset."]
        lines.append(f"Search space ({self.dim} parameters, all in [0,1] normalized):")
        for name, (lo, hi, scale) in self._space.items():
            lines.append(f"  {name}: [{lo}, {hi}] ({scale} scale)")
        lines.append("Higher CV score is better.")
        return "\n".join(lines)