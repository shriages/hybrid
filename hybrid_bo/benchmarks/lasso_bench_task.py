# hybrid_bo/benchmarks/lasso_bench_task.py
import numpy as np
from .base_task import BaseTask


class LassoBenchTask(BaseTask):
    """
    LassoBench RealBenchmark (Šehić et al.): real-data sparse-regression HPO.
    evaluate() returns cross-validation loss (lower=better) over log-alpha
    hyperparameters in [-1,1]^d; we negate so higher=better. Dimension is
    fixed by the dataset's feature count, read at construction.

    Regret reference: the oracle loss (achievable floor). In the negated
    (higher=better) space, f_star = -loss_oracle, so
    R_T = f_star - f_best = (best val_loss) - loss_oracle >= 0.
    """
    pick_data: str = None

    def __init__(self):
        from LassoBench.LassoBench import RealBenchmark
        self._bench = RealBenchmark(pick_data=self.pick_data)
        self.dim = int(self._bench.n_features)
        self.bounds = [(-1.0, 1.0)] * self.dim

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = np.clip(x, -1.0, 1.0)                 # evaluate() raises outside [-1,1]
        loss = float(self._bench.evaluate(x))     # cross-val loss, lower=better
        return -loss                              # higher=better

    def f_star(self) -> float:
        # oracle loss is the achievable floor; negated to higher=better space.
        return -float(self._bench.loss_oracle)


class LassoDiabetesTask(LassoBenchTask):
    name = "lasso_diabetes"
    pick_data = "diabetes"


class LassoBreastCancerTask(LassoBenchTask):
    name = "lasso_breast_cancer"
    pick_data = "breast_cancer"