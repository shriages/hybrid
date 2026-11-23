import numpy as np
from .base_task import BaseTask


class GallagherTask(BaseTask):
    """
    Gallagher-style multimodal function: a sum of 21 Gaussian "peaks" of
    differing heights and widths scattered across the domain. One dominant
    peak is the global optimum; the remaining 20 subordinate peaks create a
    deceptive landscape of competing local optima. Higher = better (this is a
    maximization objective by construction, so no negation is needed).

    The peak locations, heights, and widths are generated once from a FIXED
    seed and frozen as constants, so the task is fully deterministic and
    reproducible and the global optimum is known exactly: it is the center of
    the single highest-weight peak, where evaluate() attains the top height.

    f(x) = max_i  w_i * exp( -0.5 * sum_j (x_j - c_ij)^2 / s_i^2 )

    Using the MAX over peaks (rather than a sum) keeps the global optimum
    exactly equal to the dominant peak's height w_max, independent of overlap.
    """
    name = "gallagher"

    N_PEAKS = 21

    def __init__(self, dim: int = 20, seed: int = 20240101):
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self._seed = seed

        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])

        # Peak centers: uniform in the domain. Peak 0 is the global optimum;
        # place it in the interior (not on the boundary) for a clean basin.
        centers = rng.uniform(lo, hi, size=(self.N_PEAKS, dim))
        centers[0] = rng.uniform(lo * 0.5, hi * 0.5)   # dominant peak, interior

        # Heights: peak 0 is strictly the tallest (the global optimum); the
        # rest are drawn below it so none can exceed it.
        heights = rng.uniform(1.0, 8.0, size=self.N_PEAKS)
        heights[0] = 10.0                              # dominant global peak
        heights[1:] = np.minimum(heights[1:], 9.0)     # strictly below 10

        # Widths (std of each Gaussian): varied so basins differ in sharpness.
        widths = rng.uniform(0.3, 1.5, size=self.N_PEAKS)

        self._centers = centers
        self._heights = heights
        self._widths = widths

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        # squared distance from x to each peak center, shape (N_PEAKS,)
        d2 = np.sum((self._centers - x[None, :]) ** 2, axis=1)
        vals = self._heights * np.exp(-0.5 * d2 / (self._widths ** 2))
        return float(np.max(vals))

    def f_star(self) -> float:
        # The dominant peak's height, attained exactly at its center.
        return float(self._heights[0])

    def optimum_x(self) -> np.ndarray:
        """The global optimizer location (dominant peak center)."""
        return self._centers[0].copy()
