import numpy as np
from .base_task import BaseTask


class Hartmann6Task(BaseTask):
    """
    Hartmann 6D benchmark on [0,1]^6. Negated so higher = better.
    Global optimum ≈ 3.32237.
    """
    name = "hartmann6"
    dim = 6
    bounds = [(0.0, 1.0)] * 6

    _alpha = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ], dtype=float)
    _P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        inner = np.sum(self._A * (x - self._P) ** 2, axis=1)
        return float(np.sum(self._alpha * np.exp(-inner)))

    def f_star(self) -> float:
        return 3.32237