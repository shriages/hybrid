"""
OnlineConfig: config for a v2 online-switching (Exp3) run.

Separate from experiments.config.ExperimentConfig on purpose: that dataclass
validates "exactly one of epsilon / tau", which an online run has neither of.
This is a parallel, non-invasive config for the online path only.

ABLATION SUPPORT
----------------
storage_optimizer_name now also encodes:
  * the ARM ROSTER, but ONLY when it differs from the full default pool.
    This keeps every existing main-sweep path byte-identical (no re-runs),
    while giving each arm-removal ablation variant a unique path. Without
    this, "remove LLAMBO" and "full" share λ/α/η and would overwrite.
  * a SWITCH MODE tag for the deterministic one-time-switch variants
    (switch_mode="one_time_<llm_arm>"), which are not Exp3 runs at all.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


_FULL_POOL = ["gp", "llm", "llambo", "lgbo"]


@dataclass
class OnlineConfig:
    task_name: str
    llm_model: str = "deepseek-v4-flash"
    T: int = 30

    # Batch schedule
    b: int = 5
    b_first: Optional[int] = 2          # small opening batch

    # Exp3 / loss hyperparameters
    eta: float = 0.1
    lambda_: float = 0.5
    alpha: float = 0.05
    eps1: float = 1e-9
    eps2: float = 1e-9

    # Arm roster (index order defines figure colors/labels). Default = full pool.
    arms: List[str] = field(default_factory=lambda: list(_FULL_POOL))

    # Per-arm construction knobs
    n_init: int = 5
    n_candidates: int = 5

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    optimizer_name: str = "online_hybrid"
    switch_mode: Optional[str] = None
    tau: Optional[int] = None           # switch iteration; default T // 2

    _VALID_ARMS = {"gp", "tpe", "llm", "llambo", "lgbo"}

    def __post_init__(self):
        if not self.arms:
            raise ValueError("arms must be non-empty.")
        for a in self.arms:
            if a not in self._VALID_ARMS:
                raise ValueError(
                    f"Unknown arm '{a}'. Choose from {sorted(self._VALID_ARMS)}."
                )
        if len(set(self.arms)) != len(self.arms):
            raise ValueError(f"Duplicate arms not allowed: {self.arms}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0,1), got {self.alpha}.")
        if self.eta < 0.0:
            raise ValueError(f"eta must be >= 0 (eta=0 freezes weights), got {self.eta}.")

        if self.switch_mode is not None:
            if not self.switch_mode.startswith("one_time_"):
                raise ValueError(
                    f"Unknown switch_mode '{self.switch_mode}'. "
                    f"Use None or 'one_time_<llm|llambo|lgbo>'."
                )
            arm = self.switch_mode.split("one_time_", 1)[1]
            if arm not in {"llm", "llambo", "lgbo"}:
                raise ValueError(
                    f"one_time switch arm must be llm|llambo|lgbo, got '{arm}'."
                )
            tau = (self.T // 2) if self.tau is None else self.tau
            if not (0 < tau < self.T):
                raise ValueError(f"tau must satisfy 0<tau<T; got {tau} (T={self.T}).")

    # ── Convenience ─────────────────────────────────────────────────────────
    @property
    def one_time_arm(self) -> Optional[str]:
        """The LLM arm used by a one-time-switch run, else None."""
        if self.switch_mode and self.switch_mode.startswith("one_time_"):
            return self.switch_mode.split("one_time_", 1)[1]
        return None

    @property
    def effective_tau(self) -> int:
        return (self.T // 2) if self.tau is None else int(self.tau)

    @property
    def storage_optimizer_name(self) -> str:
        """ResultStore path component.

        Base encodes λ, α, η (as before, so existing main-sweep paths are
        unchanged). Additionally:
          * '_arms_a-b-c'  appended ONLY when the roster is not the full pool
            (so arm-removal ablations don't collide with the full run);
          * '_one_time_<arm>' appended for deterministic one-time-switch runs
            (which are not Exp3 at all).
        """
        base = (f"{self.optimizer_name}"
                f"_lambda_{self.lambda_:.2f}"
                f"_alpha_{self.alpha:.2f}"
                f"_eta_{self.eta:.2f}")

        if self.switch_mode:
            return f"{base}_{self.switch_mode}"

        if list(self.arms) != _FULL_POOL:
            base += "_arms_" + "-".join(self.arms)
        return base

    def as_store_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            task_name=self.task_name,
            optimizer_name=self.storage_optimizer_name,
            llm_model=self.llm_model,
        )
