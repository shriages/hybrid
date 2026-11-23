import numpy as np
from .base_task import BaseTask


class RastriginTask(BaseTask):
    """
    Rastrigin function. Strongly multimodal: a regular grid of ~10^d local
    minima surrounding a single global minimum at the origin. Negated so that
    higher = better, consistent with the rest of the suite.

    Raw:  f(x) = 10 d + sum_i [ x_i^2 - 10 cos(2 pi x_i) ],  min 0 at x = 0.
    """
    name = "rastrigin"

    def __init__(self, dim: int = 10):
        self.dim = dim
        self.bounds = [(-5.12, 5.12)] * dim

    def _raw(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    def evaluate(self, x: np.ndarray) -> float:
        return -self._raw(x)

    def f_star(self) -> float:
        return 0.0
