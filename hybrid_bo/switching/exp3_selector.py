"""
Exp3 online arm-selection for HybridBO v2 (online switching).

This module is deliberately self-contained and depends on NOTHING in the
rest of the codebase: it operates on arm *indices* and scalar losses only.
The optimizers, tasks, LLM backends, RunResult, etc. are all orchestrated
elsewhere (core/online_hybrid_optimizer.py); this file is just the bandit.

Algorithm (standard Exp3 with importance-weighted loss estimates), matching
the v2 spec:

  Initialise  W_0(a) = 1 for all a;  P_0 = uniform(N).
  Each batch j:
      sample A_j ~ P_j
      observe realized loss l_j for the PLAYED arm only
      importance-weighted estimate:  l_hat_j(a) = (l_j / P_j(a)) * 1{a = A_j}
      weight update:   W_{j+1}(a) = W_j(a) * exp(-eta * l_hat_j(a))
      prob  update:    P_{j+1}(a) = (1 - alpha) * W_{j+1}(a)/sum(W_{j+1})
                                    + alpha * (1/N)

The realized loss l_j itself (performance-vs-cost trade-off with running
maxima and the lambda preference) is computed by `compute_loss` below, kept
separate so the selector stays a generic bandit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Batch schedule: ragged first + last batch, regular `b` in between.
# ────────────────────────────────────────────────────────────────────────────

def make_batch_schedule(T: int, b: int, b_first: int | None = None) -> List[int]:
    """
    Return a list of batch widths summing exactly to T.

    - The first batch has width `b_first` (default `b`) — set it small to
      reduce the influence of the uniform-random opening pick.
    - Interior batches have width `b`.
    - The final batch absorbs the remainder (may be < b: a ragged tail).

    Examples
    --------
    make_batch_schedule(30, 8)         -> [8, 8, 8, 6]
    make_batch_schedule(30, 8, 3)      -> [3, 8, 8, 8, 3]
    make_batch_schedule(10, 4, 2)      -> [2, 4, 4]
    make_batch_schedule(5, 8, 3)       -> [3, 2]
    make_batch_schedule(3, 8)          -> [3]
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}.")
    if b_first is None:
        b_first = b
    if b_first <= 0:
        raise ValueError(f"b_first must be positive, got {b_first}.")

    if b_first >= T:
        return [T]

    schedule = [b_first]
    remaining = T - b_first
    while remaining > b:
        schedule.append(b)
        remaining -= b
    # remaining is now in [1, b]; it forms the ragged final batch.
    schedule.append(remaining)
    assert sum(schedule) == T, (schedule, T)
    print("schedule:", schedule)
    return schedule


# ────────────────────────────────────────────────────────────────────────────
# Realized per-batch loss (performance-vs-cost trade-off).
# ────────────────────────────────────────────────────────────────────────────

def compute_loss(
    I_j: float,
    C_j: float,
    I_max: float,
    C_max: float,
    lambda_: float,
    eps1: float = 1e-9,
    eps2: float = 1e-9,
) -> float:
    """
    Realized loss for the played arm in batch j (v2 spec):

        l_j = - I_j / (I_max + eps1)  +  lambda * C_j / (C_max + eps2)

    where I_j >= 0 is the batch improvement (bigger = better, hence the
    minus sign turns it into a loss), C_j >= 0 is the batch cost, and
    I_max, C_max are the running maxima over batches seen so far
    (including j). eps1/eps2 guard against a zero denominator.
    """
    unnormalized_loss = -I_j / (I_max + eps1) + lambda_ * C_j / (C_max + eps2)
    normalized_loss = (unnormalized_loss + 1.0) / (1.0 + lambda_)
    # perf_term = -I_j / (I_max + eps1)
    # cost_term = lambda_ * C_j / (C_max + eps2)
    # return perf_term + cost_term
    return normalized_loss


# ────────────────────────────────────────────────────────────────────────────
# The Exp3 selector.
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Exp3Selector:
    """
    Stateful Exp3 arm selector over `n_arms` arms.

    Parameters
    ----------
    n_arms : number of candidate optimizers N = |A|.
    eta    : learning rate for the exponential weight update.
    alpha  : uniform-mixing coefficient in (0, 1); P = (1-alpha)*w/sum(w)
             + alpha/N. Guarantees every arm keeps a floor probability
             alpha/N, which both bounds the importance weights and keeps
             exploration alive.

    State (readable for logging into the result object)
    ---------------------------------------------------
    weights : current W_j, shape (n_arms,)
    probs   : current P_j, shape (n_arms,)
    """
    n_arms: int
    eta: float = 0.1
    alpha: float = 0.05
    weights: np.ndarray = field(default=None)
    probs: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.n_arms <= 0:
            raise ValueError(f"n_arms must be positive, got {self.n_arms}.")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}.")
        self.weights = np.ones(self.n_arms, dtype=float)
        self.probs = self._weights_to_probs(self.weights)

    def _weights_to_probs(self, w: np.ndarray) -> np.ndarray:
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            # Degenerate weights (overflow/underflow) -> fall back to uniform.
            return np.full(self.n_arms, 1.0 / self.n_arms)
        return (1.0 - self.alpha) * (w / w_sum) + self.alpha / self.n_arms

    def sample(self, rng: np.random.Generator) -> int:
        """Draw an arm index A_j ~ P_j."""
        return int(rng.choice(self.n_arms, p=self.probs))

    def update(self, arm: int, realized_loss: float) -> None:
        """
        Apply one Exp3 update given the realized loss of the PLAYED arm.

        l_hat(a) = (realized_loss / P_j(arm)) * 1{a == arm}
        W(a)    <- W(a) * exp(-eta * l_hat(a))       # unplayed: *exp(0)=1
        P       <- (1-alpha)*W/sum(W) + alpha/N
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm {arm} out of range [0, {self.n_arms}).")
        
        # print("-" * 30)
        # print("selected arm:", arm)
        # print("probabilities before update:", self.probs)
        # print("weights before update:", self.weights)
        p_played = self.probs[arm]
        # p_played >= alpha/n_arms > 0 whenever alpha > 0, so this is safe.
        l_hat = realized_loss / p_played
        # Only the played arm's weight changes (others multiply by exp(0)=1).
        self.weights[arm] *= np.exp(-self.eta * l_hat)
        self.probs = self._weights_to_probs(self.weights)
        # print("intermediate l_hat:", l_hat)
        # print("probabilities after update:", self.probs)
        # print("weights after update:", self.weights)
        # print("Exp3Selector.update: arm=%d  realized_loss=%.4f  l_hat=%.4f  new_weight=%.4f  new_prob=%.4f" % (arm, realized_loss, l_hat, self.weights[arm], self.probs[arm]))

    # Convenience snapshots for logging into OnlineRunResult.
    def snapshot_probs(self) -> List[float]:
        return self.probs.tolist()

    def snapshot_weights(self) -> List[float]:
        return self.weights.tolist()