from abc import ABC, abstractmethod
import numpy as np
from .run_result import RunResult


class BaseOptimizer(ABC):
    """Abstract BO loop. Subclasses implement suggest() and observe()."""

    name: str = "base"

    def __init__(self, task, T: int, seed: int = 0):
        self.task = task
        self.T = T
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._result: RunResult = None

    @abstractmethod
    def suggest(self, t: int) -> np.ndarray:
        """Return next candidate x given current history."""
        ...

    @abstractmethod
    def observe(self, x: np.ndarray, y: float) -> None:
        """Update internal model with new (x, y) pair."""
        ...

    @abstractmethod
    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        """Return next candidate x conditioned entirely on the passed-in
        GLOBAL history. The optimizer stays stateless w.r.t. this call."""
        ...

    def step_cost(self, t: int) -> float:
        """Cost of step t in dollars. Override in LLM optimizers."""
        return 0.0

    def run(self) -> RunResult:
        self._result = RunResult(
            task_name=self.task.name,
            optimizer_name=self.name,
            T=self.T,
        )
        for t in range(self.T):
            x = self.suggest(t)
            x = self.task.clip(x)
            y = self.task.evaluate(x)
            self.observe(x, y)
            self._result.xs.append(x)
            self._result.ys.append(y)
            self._result.costs.append(self.step_cost(t))
        return self._result