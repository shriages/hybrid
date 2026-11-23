"""
OnlineHybridOptimizer: v2 online-switching driver (Exp3 over optimizer arms).

This is a SIBLING of core.hybrid_optimizer.HybridOptimizer, not a subclass.
HybridOptimizer does a one-shot LLM->GP handoff at a single tau; this does
batched Exp3 arm-selection over N optimizers with no fixed switch point.

It does NOT inherit BaseOptimizer and does NOT call any arm's run(): it
drives each arm's public suggest()/observe() directly, exactly as
BaseOptimizer.run() does internally (suggest -> task.clip -> task.evaluate
-> observe), but chooses WHICH arm runs each batch and reads the TRUE
per-step cost from each arm's CostTracker (delta around suggest()), rather
than the lossy step_cost().

Design decisions (locked with the author):
  * Only the ACTIVE arm observes its own batch; other arms are not fed the
    evaluated points. Each arm's internal model reflects only its own turns.
  * The GLOBAL f_best trajectory (all arms) drives I_j and all figures.
  * I_max, C_max are RUNNING maxima over batches seen so far (incl. j).
  * Batch schedule is ragged (small first batch, remainder-sized last).
  * Standard Exp3 importance weighting (l_hat = l / P for the played arm).
"""

from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np

from .online_run_result import OnlineRunResult
from hybrid_bo.switching.exp3_selector import (
    Exp3Selector, make_batch_schedule, compute_loss,
)


_DEFAULT_CONTEXT = (
    "You are one arm in a hybrid optimizer: several different optimizers take "
    "turns proposing points, and they all share one common evaluation history. "
    "The history below contains points proposed by ALL arms, not just you. Use "
    "the full shared history to propose the single best next point."
)

def _arm_total_cost(arm) -> float:
    """True cumulative cost of an arm so far.

    LLM arms expose a `cost_tracker` (CostTracker) whose total_cost() sums
    every recorded call. GPOptimizer has no tracker and always costs $0, so
    a missing tracker maps to 0.0. No arm is modified to support this.
    """
    tracker = getattr(arm, "cost_tracker", None)
    if tracker is None:
        return 0.0
    return float(tracker.total_cost())


class OnlineHybridOptimizer:
    name = "online_hybrid"

    def __init__(
        self,
        task,
        arms: Sequence,
        arm_names: Sequence[str],
        T: int,
        b: int,
        b_first: Optional[int] = None,
        eta: float = 0.1,
        lambda_: float = 0.5,
        alpha: float = 0.05,
        eps1: float = 1e-9,
        eps2: float = 1e-9,
        seed: int = 0,
        context: str = _DEFAULT_CONTEXT,
    ):
        if len(arms) != len(arm_names):
            raise ValueError(
                f"arms ({len(arms)}) and arm_names ({len(arm_names)}) "
                f"must have equal length."
            )
        if len(arms) == 0:
            raise ValueError("Need at least one arm.")

        self.task = task
        self.arms = list(arms)
        self.arm_names = list(arm_names)
        self.T = T
        self.b = b
        self.b_first = b_first
        self.eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.eps1 = eps1
        self.eps2 = eps2
        self.seed = seed
        self.context = context
        self.rng = np.random.default_rng(seed)

        self._result: Optional[OnlineRunResult] = None

    def run(self) -> OnlineRunResult:
        selector = Exp3Selector(
            n_arms=len(self.arms), eta=self.eta, alpha=self.alpha,
        )
        schedule = make_batch_schedule(self.T, self.b, self.b_first)

        res = OnlineRunResult(
            task_name=self.task.name,
            optimizer_name=self.name,
            T=self.T,
            arm_names=list(self.arm_names),
            lambda_=self.lambda_,
        )
        res.batch_schedule = list(schedule)

        t_global = 0
        f_best_global = float("-inf")
        I_max = 0.0
        C_max = 0.0

        global_hist: List[dict] = []
        
        for width in schedule:
            # print("Batch width:", width)
            
            # Snapshot P_j, W_j BEFORE playing this batch (what we sampled from).
            res.prob_history.append(selector.snapshot_probs())
            res.weight_history.append(selector.snapshot_weights())

            a = selector.sample(self.rng)
            arm = self.arms[a]

            f_best_before = f_best_global
            C_j = 0.0
            # print("f_best_global before batch:", f_best_before)
            for _ in range(width):

                c_before = _arm_total_cost(arm)
                # print("cost before:", c_before)

                #    Now pass the shared GLOBAL history + context so every
                #    arm conditions on ALL arms' points. ──────────────────
                x = arm.suggest_online(t_global, global_hist, self.context)

                                # x = arm.suggest(t_global)
                # print("x before clip:", x)
                x = self.task.clip(x)
                # print("x after clip:", x)
                y = float(self.task.evaluate(x))
                                # arm.observe(x, y) 
                # only the active arm observes
                c_after = _arm_total_cost(arm)
                # print("cost after:", c_after)
                step_cost = max(c_after - c_before, 0.0)
                # print("step_cost:", step_cost)
                res.xs.append(x)
                res.ys.append(y)
                res.costs.append(step_cost)
                global_hist.append({"x": x, "y": y})

                if y > f_best_global:
                    f_best_global = y
                C_j += step_cost
                # print("Cost C_j so far in this batch:", C_j)
                t_global += 1

            if f_best_before == float("-inf"):
                I_j = 0.0          # no prior best to improve on
            else:
                I_j = max(f_best_global - f_best_before, 0.0)
            # I_j = max(f_best_global - f_best_before, 0.0)
            # print("Improvement I_j:", I_j)
            I_max = max(I_max, I_j)
            # print("Improvement I_max so far:", I_max)
            C_max = max(C_max, C_j)
            # print("Cost C_max so far:", C_max)

            l_j = compute_loss(
                I_j=I_j, C_j=C_j, I_max=I_max, C_max=C_max,
                lambda_=self.lambda_, eps1=self.eps1, eps2=self.eps2,
            )
            # print("Realized loss l_j:", l_j)
            selector.update(a, l_j)

            res.batch_arms.append(a)
            res.improvements.append(I_j)
            res.batch_costs.append(C_j)
            res.losses.append(l_j)
            res.imax_history.append(I_max)
            res.cmax_history.append(C_max)

        self._result = res
        return res