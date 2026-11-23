import numpy as np
from .base_task import BaseTask


class SchwefelTask(BaseTask):
    """
    Schwefel 2.26 function. Deceptive multimodal: the global minimum sits far
    from the next-best local minima and near the domain boundary, so greedy
    local search is systematically misled away from it. Negated so higher =
    better.

    Raw:  f(x) = 418.9829 d - sum_i x_i sin(sqrt(|x_i|)),
    global min ~ 0 at x_i = 420.9687 for all i.
    """
    name = "schwefel"

    def __init__(self, dim: int = 14):
        self.dim = dim
        self.bounds = [(-500.0, 500.0)] * dim

    def _raw(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def evaluate(self, x: np.ndarray) -> float:
        return -self._raw(x)

    def f_star(self) -> float:
        # Raw minimum is ~0 (a tiny positive residual from the 418.9829
        # constant vs. the exact optimum); the negated optimum is ~0.
        return 0.0
