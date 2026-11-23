"""
OnlineRunResult: the output object for one v2 online-switching (Exp3) run.

It is a SIBLING of core.run_result.RunResult, not a subclass and not an
edit of it. It intentionally exposes the same four read-only properties
(`f_best`, `f_best_curve`, `cost_curve`, `total_cost`) so that any existing
figure/metric code written against RunResult's duck-type keeps working when
handed an OnlineRunResult — without either class depending on the other.

Layout
------
Global per-iteration streams (length T after a run), across ALL arms:
    xs, ys, costs                # costs = true CostTracker-delta per step
Per-batch Exp3 bookkeeping (length = number of batches):
    batch_schedule               # width of each batch; sums to T
    batch_arms                   # A_j: arm index played in batch j
    prob_history                 # P_j (BEFORE playing batch j), each len n_arms
    weight_history               # W_j (BEFORE playing batch j), each len n_arms
    improvements                 # I_j: global f_best gain during batch j (>= 0)
    batch_costs                  # C_j: true cost incurred during batch j
    losses                       # l_j: realized loss fed to Exp3 for A_j
    imax_history, cmax_history   # running maxima used to normalise l_j
Static metadata:
    task_name, optimizer_name, T, arm_names, lambda_
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class OnlineRunResult:
    task_name: str
    optimizer_name: str
    T: int

    # Static config echoed for downstream figures/tables.
    arm_names: List[str] = field(default_factory=list)   # index -> name
    lambda_: Optional[float] = None                      # the λ this run used

    # ── Global per-iteration streams (all arms), length T after run ──────────
    xs: List[np.ndarray] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)     # true per-step cost

    # ── Per-batch Exp3 bookkeeping, length = n_batches after run ─────────────
    batch_schedule: List[int] = field(default_factory=list)
    batch_arms: List[int] = field(default_factory=list)
    prob_history: List[List[float]] = field(default_factory=list)
    weight_history: List[List[float]] = field(default_factory=list)
    improvements: List[float] = field(default_factory=list)
    batch_costs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    imax_history: List[float] = field(default_factory=list)
    cmax_history: List[float] = field(default_factory=list)

    # ── Duck-typed to match RunResult (figures rely on these names) ──────────
    @property
    def f_best_curve(self) -> List[float]:
        """Cumulative best observed value at each global step."""
        curve, best = [], float("-inf")
        for y in self.ys:
            best = max(best, y)
            curve.append(best)
        return curve

    @property
    def cost_curve(self) -> List[float]:
        """Cumulative true cost at each global step."""
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

    # ── v2-specific derived views (for Figures 1–6) ──────────────────────────
    @property
    def n_arms(self) -> int:
        return len(self.arm_names)

    @property
    def n_batches(self) -> int:
        return len(self.batch_arms)

    @property
    def batch_boundaries(self) -> List[int]:
        """Cumulative global-iteration index at the end of each batch.
        schedule [3,8,8,8,3] -> [3,11,19,27,30]. Maps per-iter streams to j."""
        bounds, acc = [], 0
        for w in self.batch_schedule:
            acc += w
            bounds.append(acc)
        return bounds

    @property
    def prob_matrix(self) -> np.ndarray:
        """(n_batches, n_arms) array of P_j — Figure 1 (probability over batches)."""
        if not self.prob_history:
            return np.zeros((0, self.n_arms))
        return np.asarray(self.prob_history, dtype=float)

    @property
    def final_arm(self) -> Optional[int]:
        """Arm played in the last batch — Figure 3 'final dominant method'."""
        return self.batch_arms[-1] if self.batch_arms else None

    @property
    def final_arm_name(self) -> Optional[str]:
        idx = self.final_arm
        if idx is None or not self.arm_names:
            return None
        return self.arm_names[idx]

    def transition_counts(self) -> np.ndarray:
        """(n_arms, n_arms) matrix M[a,b] = # of batch j where A_j=a, A_{j+1}=b.
        Figure 6 (transition heatmap). Rows are 'from', columns 'to'."""
        n = self.n_arms
        M = np.zeros((n, n), dtype=float)
        for a, b in zip(self.batch_arms[:-1], self.batch_arms[1:]):
            M[a, b] += 1.0
        return M

    def arm_selection_counts(self) -> np.ndarray:
        """(n_arms,) count of batches each arm was played (Figure 2/3 support)."""
        counts = np.zeros(self.n_arms, dtype=float)
        for a in self.batch_arms:
            counts[a] += 1.0
        return counts