import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .base_optimizer import BaseOptimizer


class GPOptimizer(BaseOptimizer):
    """Pure GP-BO with Expected Improvement acquisition."""

    name = "gp"

    def __init__(self, task, T: int, n_init: int = 1, seed: int = 0):
        super().__init__(task, T, seed)
        self.n_init = n_init
        self._X: list = []
        self._y: list = []
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=seed,
        )

    def _ei(self, X_cand: np.ndarray) -> np.ndarray:
        """Expected Improvement over current best."""
        if len(self._y) == 0:
            return np.zeros(len(X_cand))
        mu, sigma = self._gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        f_best = max(self._y)
        Z = (mu - f_best) / sigma
        return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

    def _random_candidates(self, n: int = 1000) -> np.ndarray:
        lo = np.array([b[0] for b in self.task.bounds])
        hi = np.array([b[1] for b in self.task.bounds])
        return self.rng.uniform(lo, hi, size=(n, self.task.dim))

    def suggest(self, t: int) -> np.ndarray:
        if t < self.n_init or len(self._X) < self.n_init:
            return self.task.random_x(self.rng)
        X_cand = self._random_candidates()
        ei = self._ei(X_cand)
        return X_cand[np.argmax(ei)]

    def observe(self, x: np.ndarray, y: float) -> None:
        self._X.append(x)
        self._y.append(y)
        if len(self._X) >= self.n_init:
            self._gp.fit(np.array(self._X), np.array(self._y))
            
    @staticmethod
    def _ei_from(gp, ys, X_cand: np.ndarray) -> np.ndarray:
        if len(ys) == 0:
            return np.zeros(len(X_cand))
        mu, sigma = gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        f_best = max(ys)
        Z = (mu - f_best) / sigma
        return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)


    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        if len(history) < self.n_init:
            return self.task.random_x(self.rng)
 
        X = np.array([np.asarray(h["x"], dtype=float) for h in history])
        y = [float(h["y"]) for h in history]
 
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=self.seed,
        )
        gp.fit(X, np.array(y))
 
        X_cand = self._random_candidates()
        ei = self._ei_from(gp, y, X_cand)
        return X_cand[np.argmax(ei)]
