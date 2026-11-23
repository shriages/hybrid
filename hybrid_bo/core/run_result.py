from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RunResult:
    """Holds everything produced by one optimizer run."""
    task_name: str
    optimizer_name: str
    T: int                                  # total iterations
    tau: Optional[int] = None              # switching time (None if no switching)

    # Per-iteration history (length T after run)
    xs: List[np.ndarray] = field(default_factory=list)   # candidates
    ys: List[float] = field(default_factory=list)         # observations
    costs: List[float] = field(default_factory=list)      # per-step LLM cost ($)
    d_ts: List[float] = field(default_factory=list)       # distance metric per step

    @property
    def f_best_curve(self) -> List[float]:
        """Cumulative best observed value at each step."""
        curve, best = [], float("-inf")
        for y in self.ys:
            best = max(best, y)
            curve.append(best)
        return curve

    @property
    def cost_curve(self) -> List[float]:
        """Cumulative cost at each step."""
        total, curve = 0.0, []
        for c in self.costs:
            total += c
            curve.append(total)
        return curve

    @property
    def total_cost(self) -> float:
        return sum(self.costs)

    @property
    def f_best(self) -> float:
        return max(self.ys) if self.ys else float("-inf")