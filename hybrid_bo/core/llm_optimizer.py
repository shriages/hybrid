"""
LLMOptimizer: pure LLM baseline for Bayesian optimization (no EI, no GP —
the LLM directly proposes the next candidate at every step).

Behavior:
  - Step 0 (no observations yet): the LLM is asked to propose an initial
    point using only the task description (no history section at all).
  - Step t > 0: the LLM is shown the full observed trajectory so far and
    asked to propose the next point based on it.

This mirrors the LGBOOptimizer refactor: the former `LLMPromptBuilder` is
merged directly into this class as `_build_prompt`/`_format_history`, and
all external touchpoints are preserved: `LLMOptimizer(task, T, llm, seed=...)`,
`.name`, `.suggest(t)`, `.observe(x, y)`, `.step_cost(t)`, `.cost_tracker`.
"""

import json
import numpy as np
from .base_optimizer import BaseOptimizer
# from hybrid_bo.utils.extract_json import _extract_json
from hybrid_bo.llm_backends.cost_tracker import CostTracker
from hybrid_bo.benchmarks.task_metadata import describe_task


class LLMOptimizer(BaseOptimizer):
    """
    Pure LLM optimizer: at each step, the LLM proposes the next candidate x.
    Falls back to random suggestion if the LLM response cannot be parsed.
    """

    name = "llm"

    def __init__(self, task, T: int, llm, seed: int = 0):
        """
        Args:
            task:  any BaseTask instance
            T:     total iterations
            llm:   a BaseLLM instance (OpenAILLM, ClaudeLLM, etc.)
            seed:  random seed for fallback
        """
        super().__init__(task, T, seed)
        self._llm = llm
        self._tracker = CostTracker()
        self._history = []          # list of {"x": np.ndarray, "y": float}
        self._last_cost = 0.0       # cost of most recent LLM call

    # ------------------------------------------------------------------ #
    # Prompt construction (formerly LLMPromptBuilder)
    # ------------------------------------------------------------------ #
    def _task_description(self) -> str:
        # Task-specific semantic description (objective/direction, per-variable
        # meanings, real ranges/scales, and — for real experiments — physical
        # background), resolved from the task's `.name`. The optimizer still
        # proposes/parses raw coordinate vectors of length task.dim in the same
        # space evaluate() expects; describe_task() only enriches the wording,
        # so the {"x": [...]} JSON contract below is unchanged.
        return describe_task(self.task)

    @staticmethod
    def _format_history(history, max_entries: int = 20) -> str:
        if not history:
            return "No observations yet."
        rows = []
        for i, h in enumerate(history[-max_entries:]):
            x_str = ", ".join(f"{v:.4f}" for v in h["x"])
            rows.append(f"  [{i+1}] x=[{x_str}]  y={h['y']:.6f}")
        return "\n".join(rows)

    def _build_prompt(self) -> str:
        task_description = self._task_description()
        
        # ── First proposal: no history at all ──────────────────────────
        if not self._history:
            import hashlib
            hint_seed = int(hashlib.md5(f"llm_{self.rng}".encode()).hexdigest()[:8], 16)
            hint_rng = np.random.default_rng(hint_seed)
            focus = hint_rng.uniform(
                [b[0] for b in self.task.bounds],
                [b[1] for b in self.task.bounds],
            )
            focus_str = ", ".join(f"{v:.3f}" for v in focus)
            return (
                f"You are a Bayesian optimization assistant.\n\n"
                f"Task:\n{task_description}\n\n"
                f"No points have been evaluated yet. Begin your search near "
                f"[{focus_str}], then adjust using your best judgment about "
                f"where a promising point might be within the bounds.\n"
                f"Return ONLY a JSON object: {{\"x\": [v1, v2, ...]}}. "
                f"No explanation."
            )

        # ── Subsequent proposals: condition on observed trajectory ─────
        history_str = self._format_history(self._history)
        return (
            f"You are a Bayesian optimization assistant.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Observed points so far:\n{history_str}\n\n"
            f"Based on this history, propose the single best next point to "
            f"evaluate. "
            f"Return ONLY a JSON object: {{\"x\": [v1, v2, ...]}}. "
            f"No explanation."
        )
        
    def online_prompt_builder(self, history, context: str = "") -> str:
        task_description = self._task_description()
        ctx = f"{context}\n\n" if context else ""
 
        if not history:
            import hashlib
            hint_seed = int(hashlib.md5(f"llm_{self.rng}".encode()).hexdigest()[:8], 16)
            hint_rng = np.random.default_rng(hint_seed)
            focus = hint_rng.uniform(
                [b[0] for b in self.task.bounds],
                [b[1] for b in self.task.bounds],
            )
            focus_str = ", ".join(f"{v:.3f}" for v in focus)
            return (
                f"{ctx}"
                f"You are a Bayesian optimization assistant.\n\n"
                f"Task:\n{task_description}\n\n"
                f"No points have been evaluated yet. Begin your search near "
                f"[{focus_str}], then adjust using your best judgment about "
                f"where a promising point might be within the bounds.\n"
                f"Return ONLY a JSON object: {{\"x\": [v1, v2, ...]}}. "
                f"No explanation."
            )
 
        history_str = self._format_history(history)
        return (
            f"{ctx}"
            f"You are a Bayesian optimization assistant.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Observed points so far:\n{history_str}\n\n"
            f"Based on this history, propose the single best next point to "
            f"evaluate. "
            f"Return ONLY a JSON object: {{\"x\": [v1, v2, ...]}}. "
            f"No explanation."
        )


    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def _parse(self, text: str) -> np.ndarray:
        """Extract x array from LLM JSON response. Returns None on failure."""
        try:
            obj = json.loads(text.strip())
            x = np.array(obj["x"], dtype=float)
            if x.shape == (self.task.dim,):
                return x
        except Exception:
            pass
        return None
    
    # ------------------------------------------------------------------ #
    # BaseOptimizer interface
    # ------------------------------------------------------------------ #
    def suggest(self, t: int) -> np.ndarray:
        prompt = self._build_prompt()
        response = self._llm.complete(prompt, max_tokens=128, json_only=True)
        self._last_cost = response["cost"]
        self._tracker.record(
            t=t,
            model=self._llm.model,
            optimizer=self.name,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=response["cost"],
        )
        x = self._parse(response["text"])
        if x is None:
            # Fallback: random point (still records the cost)
            x = self.task.random_x(self.rng)
        return x

    def observe(self, x: np.ndarray, y: float) -> None:
        self._history.append({"x": x, "y": y})
        
    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        prompt = self.online_prompt_builder(history, context)
        response = self._llm.complete(prompt, max_tokens=128, json_only=True)
        self._last_cost = response["cost"]
        self._tracker.record(
            t=t,
            model=self._llm.model,
            optimizer=self.name,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=response["cost"],
        )
        x = self._parse(response["text"])
        if x is None:
            x = self.task.random_x(self.rng)
        return x


    def step_cost(self, t: int) -> float:
        # return self._last_cost
        return self._tracker.cost_at(t)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._tracker