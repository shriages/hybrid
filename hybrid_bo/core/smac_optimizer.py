"""
SMACOptimizer: SMAC3 baseline (random-forest surrogate BO).

Uses SMAC3's ask-and-tell interface, which maps onto BaseOptimizer's
suggest()/observe() contract. SMAC is the third classical-BO reference point
alongside GP-BO (Gaussian-process surrogate) and TPE (Parzen estimator):
it uses a RANDOM FOREST surrogate, which is what most AutoML systems deploy.

Cost: SMAC makes NO LLM calls, so it is free ($0.00), like GP and TPE.

IMPORTANT — dependency conflict:
    SMAC 2.x requires ConfigSpace >= 1.0, while yahpo_gym 1.0.2 pins
    ConfigSpace <= 0.6.1. They cannot coexist in one environment, so SMAC
    runs on every task EXCEPT the yahpo_* ones. This is enforced by giving
    SMAC its own Modal image that does not install yahpo_gym.

SMAC MINIMIZES by default; our convention is MAXIMIZE (higher evaluate() is
better), so the objective handed to SMAC is negated on the way in and the
value we record is negated back on the way out.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .base_optimizer import BaseOptimizer


class SMACOptimizer(BaseOptimizer):
    """SMAC3 (random-forest BO), ask/tell. No LLM, no cost."""

    name = "smac"

    def __init__(self, task, T: int, n_init: int = 5, seed: int = 0,
                 output_dir: Optional[str] = None):
        super().__init__(task, T, seed)
        self.n_init = n_init

        from ConfigSpace import ConfigurationSpace, Float
        from smac import HyperparameterOptimizationFacade as HPOFacade
        from smac import Scenario

        self._Float = Float

        # Build a flat continuous space matching the task's real bounds.
        self._param_names = [f"x{i}" for i in range(task.dim)]
        cs = ConfigurationSpace(seed=seed)
        for name, (lo, hi) in zip(self._param_names, task.bounds):
            cs.add(Float(name, (float(lo), float(hi))))
        self._cs = cs

        # SMAC writes run history to disk; keep it in a unique temp dir per run
        # so parallel seeds/tasks never collide.
        import tempfile
        self._outdir = output_dir or tempfile.mkdtemp(prefix="smac_")

        scenario = Scenario(
            cs,
            n_trials=T,
            deterministic=True,
            seed=seed,
            output_directory=self._outdir,
        )

        # A dummy target function: we never let SMAC drive the loop (we use
        # ask/tell), but the facade requires a callable at construction.
        def _dummy(config, seed: int = 0) -> float:
            return 0.0

        self._smac = HPOFacade(
            scenario,
            _dummy,
            initial_design=HPOFacade.get_initial_design(scenario, n_configs=n_init),
            overwrite=True,
        )

        self._pending = {}          # t -> TrialInfo awaiting tell()
        self._last_t = None

    # ── helpers ─────────────────────────────────────────────────────────────
    def _config_to_x(self, config) -> np.ndarray:
        return np.array([float(config[n]) for n in self._param_names], dtype=float)

    def _x_to_config(self, x: np.ndarray):
        from ConfigSpace import Configuration
        values = {n: float(v) for n, v in zip(self._param_names, np.asarray(x, float))}
        return Configuration(self._cs, values=values)

    # ── BaseOptimizer interface ─────────────────────────────────────────────
    def suggest(self, t: int) -> np.ndarray:
        info = self._smac.ask()
        self._pending[t] = info
        self._last_t = t
        return self._config_to_x(info.config)

    def observe(self, x: np.ndarray, y: float) -> None:
        from smac.runhistory import TrialValue

        info = self._pending.pop(self._last_t, None)
        if info is None:
            return                       # out of step; nothing to tell
        # SMAC minimizes -> hand it the negated objective.
        self._smac.tell(info, TrialValue(cost=-float(y)))

    def step_cost(self, t: int) -> float:
        return 0.0                       # SMAC makes no LLM calls

    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        """Stateless variant: rebuild a SMAC run from the SHARED global history,
        then ask for the next point. Mirrors GPOptimizer/TPEOptimizer, which
        refit on the passed-in history rather than their own internal state."""
        if len(history) < self.n_init:
            return self.task.random_x(self.rng)

        import tempfile
        from smac import HyperparameterOptimizationFacade as HPOFacade
        from smac import Scenario
        from smac.runhistory import TrialInfo, TrialValue

        scenario = Scenario(
            self._cs,
            n_trials=max(self.T, len(history) + 1),
            deterministic=True,
            seed=self.seed,
            output_directory=tempfile.mkdtemp(prefix="smac_online_"),
        )

        def _dummy(config, seed: int = 0) -> float:
            return 0.0

        smac = HPOFacade(
            scenario, _dummy,
            initial_design=HPOFacade.get_initial_design(scenario, n_configs=self.n_init),
            overwrite=True,
        )

        # Replay the shared history into SMAC's run history.
        for h in history:
            try:
                cfg = self._x_to_config(np.asarray(h["x"], dtype=float))
                smac.tell(TrialInfo(config=cfg, seed=self.seed),
                          TrialValue(cost=-float(h["y"])),
                          save=False)
            except Exception:
                continue                 # skip points outside the space

        info = smac.ask()
        return self._config_to_x(info.config)