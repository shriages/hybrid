import numpy as np
from .base_task import BaseTask


class BraninTask(BaseTask):
    """
    Branin 2D benchmark. Negated so higher = better.
    Global optimum ≈ -0.397887 at three known locations.
    """
    name = "branin"
    dim = 2
    bounds = [(-5.0, 10.0), (0.0, 15.0)]

    _a = 1.0
    _b = 5.1 / (4 * np.pi ** 2)
    _c = 5.0 / np.pi
    _r = 6.0
    _s = 10.0
    _t = 1.0 / (8 * np.pi)
    X_STARS = np.array([
        [-np.pi,  12.275],
        [ np.pi,   2.275],
        [ 9.42478, 2.47500],
    ])
    def _raw(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        t1 = self._a * (x2 - self._b * x1 ** 2 + self._c * x1 - self._r) ** 2
        t2 = self._s * (1 - self._t) * np.cos(x1)
        return float(t1 + t2 + self._s)

    def evaluate(self, x: np.ndarray) -> float:
        return -self._raw(x)  # using negative to find maximum

    def f_star(self) -> float:
        return -0.397887