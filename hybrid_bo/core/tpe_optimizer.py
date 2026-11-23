"""
TPEOptimizer: Tree-structured Parzen Estimator baseline.

Uses Optuna's TPESampler through its ASK-AND-TELL API, which maps exactly onto
BaseOptimizer's suggest()/observe() contract — no closed-loop objective, no
threads, no inversion of control.

Cost: TPE makes NO LLM calls, so it is free ($0.00), like GP-BO. It is the
"strong classical BO" reference point: a well-established, sample-efficient
optimizer that costs nothing, against which any paid LLM assistance must
justify itself.

Optuna is asked for a value per dimension in the task's real bounds, so the
same [lo, hi] box the other optimizers see is what TPE searches. Optuna
minimizes by default; our convention is MAXIMIZE (higher evaluate() = better),
so the study is created with direction="maximize" and observed y is passed
through unchanged.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from .base_optimizer import BaseOptimizer


class TPEOptimizer(BaseOptimizer):
    """Optuna TPE (ask/tell). No LLM, no cost."""

    name = "tpe"

    def __init__(self, task, T: int, n_init: int = 5, seed: int = 0):
        super().__init__(task, T, seed)
        self.n_init = n_init

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # silence per-trial logs

        self._optuna = optuna
        # n_startup_trials: TPE samples randomly until it has this many points,
        # then switches to the Parzen-estimator model. Aligning it with n_init
        # keeps the warm-up comparable to the GP arm's random warm-up.
        self._sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_init,
        )
        self._study = optuna.create_study(
            direction="maximize",           # our convention: higher = better
            sampler=self._sampler,
        )
        self._param_names = [f"x{i}" for i in range(task.dim)]
        self._pending = {}                  # t -> optuna Trial awaiting tell()

    # ── BaseOptimizer interface ─────────────────────────────────────────────
    def suggest(self, t: int) -> np.ndarray:
        trial = self._study.ask()
        x = np.array([
            trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in zip(self._param_names, self.task.bounds)
        ], dtype=float)
        self._pending[t] = trial
        self._last_t = t
        return x

    def observe(self, x: np.ndarray, y: float) -> None:
        # Tell Optuna the value of the most recent asked trial.
        t = getattr(self, "_last_t", None)
        trial = self._pending.pop(t, None)
        if trial is None:
            # Defensive: if suggest/observe got out of step, register the point
            # as a completed trial anyway so the model still sees it.
            self._study.add_trial(
                self._optuna.trial.create_trial(
                    params={n: float(v) for n, v in zip(self._param_names, x)},
                    distributions={
                        n: self._optuna.distributions.FloatDistribution(lo, hi)
                        for n, (lo, hi) in zip(self._param_names, self.task.bounds)
                    },
                    value=float(y),
                )
            )
            return
        self._study.tell(trial, float(y))

    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        """Stateless variant: rebuild a TPE study from the SHARED global history,
        then ask for the next point. Mirrors GPOptimizer.suggest_online, which
        refits its GP on the passed-in history rather than its own state."""
        if len(history) < self.n_init:
            return self.task.random_x(self.rng)

        sampler = self._optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=self.n_init,
        )
        study = self._optuna.create_study(direction="maximize", sampler=sampler)

        dists = {
            n: self._optuna.distributions.FloatDistribution(lo, hi)
            for n, (lo, hi) in zip(self._param_names, self.task.bounds)
        }
        for h in history:
            xv = np.asarray(h["x"], dtype=float)
            study.add_trial(
                self._optuna.trial.create_trial(
                    params={n: float(v) for n, v in zip(self._param_names, xv)},
                    distributions=dists,
                    value=float(h["y"]),
                )
            )

        trial = study.ask()
        return np.array([
            trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in zip(self._param_names, self.task.bounds)
        ], dtype=float)

    def step_cost(self, t: int) -> float:
        return 0.0          # TPE makes no LLM calls