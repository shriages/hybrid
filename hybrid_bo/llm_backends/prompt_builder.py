"""
One PromptBuilder per optimizer type. Each optimizer uses LLM differently,
so prompt length (and therefore cost) differs even for the same trajectory.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BasePromptBuilder(ABC):
    """Builds the prompt string and estimates its token cost before sending."""

    @abstractmethod
    def build_prompt(self, task_description: str, history: List[Dict[str, Any]]) -> str:
        """
        Assemble the full prompt string.

        Args:
            task_description: natural language description of the task and bounds.
            history: list of {"x": np.ndarray, "y": float} observed so far.

        Returns:
            Complete prompt string ready to send to the LLM.
        """
        ...

    def format_history(self, history: List[Dict[str, Any]], max_entries: int = 20) -> str:
        """Shared utility: format observed (x, y) pairs as a readable table."""
        if not history:
            return "No observations yet."
        rows = []
        for i, h in enumerate(history[-max_entries:]):
            x_str = ", ".join(f"{v:.4f}" for v in h["x"])
            rows.append(f"  [{i+1}] x=[{x_str}]  y={h['y']:.6f}")
        return "\n".join(rows)


class LLMPromptBuilder(BasePromptBuilder):
    """
    For LLMOptimizer: LLM proposes the next candidate directly.
    Prompt = task description + observed history + instruction to propose next x.
    Token usage: moderate — grows linearly with history length.
    """

    def build_prompt(self, task_description: str, history: List[Dict[str, Any]]) -> str:
        history_str = self.format_history(history)
        return (
            f"You are a Bayesian optimization assistant.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Observed points so far:\n{history_str}\n\n"
            f"Propose the single best next point to evaluate. "
            f"Return ONLY a JSON object: {{\"x\": [v1, v2, ...]}}. "
            f"No explanation."
        )


class LLAMBOPromptBuilder(BasePromptBuilder):
    """
    For LLAMBOOptimizer: LLM scores a candidate pool (surrogate role).
    Prompt = task description + candidate pool + scoring instruction.
    Token usage: HIGH — includes full candidate pool, grows with pool size.
    """

    def __init__(self, n_candidates: int = 20):
        self.n_candidates = n_candidates

    def build_prompt(
        self,
        task_description: str,
        history: List[Dict[str, Any]],
        candidates: List[Any] = None,
    ) -> str:  # type: ignore[override]
        history_str = self.format_history(history)
        cand_str = ""
        if candidates:
            lines = []
            for i, c in enumerate(candidates):
                x_str = ", ".join(f"{v:.4f}" for v in c)
                lines.append(f"  Candidate {i}: [{x_str}]")
            cand_str = "\n".join(lines)
        return (
            f"You are a surrogate model for Bayesian optimization.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Observed points:\n{history_str}\n\n"
            f"Rank the following candidates by predicted objective value (best first).\n"
            f"Candidates:\n{cand_str}\n\n"
            f"Return ONLY a JSON array of candidate indices in ranked order: [i, j, k, ...]."
        )


class LGBOPromptBuilder(BasePromptBuilder):
    """
    For LGBOOptimizer: LLM continuously injects semantic preference.
    Prompt = task description + full trajectory + semantic preference request.
    Token usage: HIGHEST — includes full trajectory at every step.
    """

    def build_prompt(self, task_description: str, history: List[Dict[str, Any]]) -> str:
        history_str = self.format_history(history, max_entries=50)
        return (
            f"You are a domain expert guiding a Bayesian optimizer.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Full optimization trajectory so far:\n{history_str}\n\n"
            f"Based on observed patterns and domain knowledge, describe which "
            f"region of the search space looks most promising. "
            f"Return ONLY a JSON object: "
            f"{{\"preferred_region\": {{\"x_min\": [...], \"x_max\": [...]}}, "
            f"\"reasoning\": \"one sentence\"}}."
        )