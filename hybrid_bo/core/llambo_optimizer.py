import json
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from .base_optimizer import BaseOptimizer
from hybrid_bo.llm_backends.cost_tracker import CostTracker
from hybrid_bo.benchmarks.task_metadata import describe_task


class LLAMBOOptimizer(BaseOptimizer):
    """
    LLAMBO-style optimizer (Liu et al., 2024), single-class implementation.

    Unlike a GP-BO hybrid, LLAMBO replaces BOTH halves of standard BO with
    the LLM:
      1. Candidate sampler (generative role): the LLM proposes candidate
         points conditioned on a target/desired objective value, using
         few-shot examples drawn from observed history. This stands in for
         random/Sobol candidate generation.
      2. Discriminative surrogate (regression role): the LLM predicts the
         objective value for each candidate via few-shot prompting, queried
         `n_gens` times per candidate to obtain a predictive mean/std --
         this stands in for the GP posterior mean/std.
    Expected Improvement is then computed analytically from the LLM
    surrogate's (mu, sigma), exactly as it would be from a GP. No GP is
    used anywhere in this optimizer.

    Note: this makes ~n_candidates * n_gens LLM calls per suggest() in the
    main loop (plus 1 for sampling) -- true to the paper's premise but far
    more expensive per iteration than a GP-based method.
    """

    name = "llambo"

    def __init__(self, task, T: int, llm,
                 n_init: int = 0, n_candidates: int = 5,
                 n_gens: int = 2, seed: int = 0):
        super().__init__(task, T, seed)
        self._llm = llm
        self._tracker = CostTracker()
        self._history: List[Dict[str, Any]] = []
        self._last_cost = 0.0
        # Kept for external/back-compat access (e.g. logging, result
        # tables). NOT used to gate control flow: every suggested point,
        # including the very first, goes through the same LLAMBO
        # candidate-sample + surrogate-score loop. The only real branch is
        # "do we have any history yet to condition on."
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.n_gens = n_gens
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        print(f"[LLAMBO INIT] n_candidates={n_candidates} n_gens={n_gens}")

    # ───────────────────────────── prompt building ────────────────────────

    def _task_description(self) -> str:
        # Task-specific semantic description resolved from the task's `.name`
        # (objective/direction, per-variable meanings, real ranges/scales, and
        # real-experiment background). Both LLAMBO roles — the generative
        # candidate sampler and the discriminative surrogate — embed this same
        # description, so grounding it in task semantics improves both. The
        # candidate/point/scalar JSON contracts and the raw-coordinate parsing
        # are unchanged; only the descriptive wording is enriched.
        return describe_task(self.task)

    def _format_history(self) -> str:
        if not self._history:
            return "  (none yet)"
        lines = []
        for h in self._history:
            x_str = ", ".join(f"{v:.4f}" for v in h["x"])
            lines.append(f"  x=[{x_str}] -> y={h['y']:.6f}")
        return "\n".join(lines)

    def _target_value(self) -> float:
        """
        Desired objective value to condition candidate generation on,
        mirroring LLAMBO's "recommend a configuration that achieves target
        performance of {jittered_desired_fval}" prompt (candidate sampler,
        Sec 3.2). We push slightly above the best observed value with
        random jitter to bias generation toward improvement while still
        varying the ask across iterations.
        """
        f_best = max(self._y)
        spread = float(np.std(self._y)) + 1e-6
        jitter = self.rng.uniform(0.05, 0.25) * spread
        return f_best + jitter

    def _build_sampler_prompt(self, n_points: int) -> str:
        desc = self._task_description()
        if not self._y:
            goal = (
                f"No observations yet. Propose {n_points} diverse, "
                f"promising points to evaluate first."
            )
        else:
            target = self._target_value()
            hist = self._format_history()
            goal = (
                f"Observed points so far:\n{hist}\n\n"
                f"Propose {n_points} NEW candidate points (not already "
                f"observed) that you believe could achieve an objective "
                f"value near {target:.6f}, i.e. better than the current "
                f"best. Favor diversity across the {n_points} proposals."
            )
        return (
            f"{desc}\n\n{goal}\n\n"
            f"Return ONLY JSON: {{\"points\": [[v1, v2, ...], ...]}} "
            f"with exactly {n_points} points, each of length {self.task.dim}."
        )

    def _build_surrogate_prompt(self, candidate: np.ndarray) -> str:
        desc = self._task_description()
        hist = self._format_history()
        x_str = ", ".join(f"{v:.4f}" for v in candidate)
        return (
            f"{desc}\n\n"
            f"Act as a regression surrogate model. Given the observed "
            f"points and their objective values below, predict the "
            f"objective value at a new point.\n\n"
            f"Observed points:\n{hist}\n\n"
            f"New point: x=[{x_str}]\n\n"
            f"Return ONLY JSON: {{\"y\": predicted_value}}."
        )
        
    @staticmethod
    def _format_history_list(history) -> str:
        if not history:
            return "  (none yet)"
        lines = []
        for h in history:
            x_str = ", ".join(f"{v:.4f}" for v in h["x"])
            lines.append(f"  x=[{x_str}] -> y={h['y']:.6f}")
        return "\n".join(lines)
 
    def _target_value_from(self, ys) -> float:
        f_best = max(ys)
        spread = float(np.std(ys)) + 1e-6
        jitter = self.rng.uniform(0.05, 0.25) * spread
        return f_best + jitter
 
    def _build_online_sampler_prompt(self, n_points: int, history, context: str = "") -> str:
        desc = self._task_description()
        ctx = f"{context}\n\n" if context else ""
        ys = [h["y"] for h in history]
        if not ys:
            goal = (
                f"No observations yet. Propose {n_points} diverse, "
                f"promising points to evaluate first."
            )
        else:
            target = self._target_value_from(ys)
            hist = self._format_history_list(history)
            goal = (
                f"Observed points so far:\n{hist}\n\n"
                f"Propose {n_points} NEW candidate points (not already "
                f"observed) that you believe could achieve an objective "
                f"value near {target:.6f}, i.e. better than the current "
                f"best. Favor diversity across the {n_points} proposals."
            )
        return (
            f"{ctx}{desc}\n\n{goal}\n\n"
            f"Return ONLY JSON: {{\"points\": [[v1, v2, ...], ...]}} "
            f"with exactly {n_points} points, each of length {self.task.dim}."
        )
 
    def _build_online_surrogate_prompt(self, candidate: np.ndarray, history, context: str = "") -> str:
        desc = self._task_description()
        ctx = f"{context}\n\n" if context else ""
        hist = self._format_history_list(history)
        x_str = ", ".join(f"{v:.4f}" for v in candidate)
        return (
            f"{ctx}{desc}\n\n"
            f"Act as a regression surrogate model. Given the observed "
            f"points and their objective values below, predict the "
            f"objective value at a new point.\n\n"
            f"Observed points:\n{hist}\n\n"
            f"New point: x=[{x_str}]\n\n"
            f"Return ONLY JSON: {{\"y\": predicted_value}}."
        )

    # ───────────────────────────── LLM call + tracking ────────────────────

    def _call_llm(self, t: int, prompt: str, max_tokens: int) -> dict:
        response = self._llm.complete(prompt, max_tokens=max_tokens)
        self._last_cost = response["cost"]
        self._tracker.record(
            t=t, model=self._llm.model, optimizer=self.name,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=response["cost"],
        )
        return response

    # ───────────────────────────── parsing ─────────────────────────────────

    def _parse_points(self, text: str, n_expected: int) -> Optional[List[np.ndarray]]:
        try:
            obj = json.loads(text.strip())
            out = []
            for p in obj["points"]:
                x = np.array(p, dtype=float)
                if x.shape == (self.task.dim,):
                    out.append(x)
            return out or None
        except Exception:
            return None

    def _parse_scalar(self, text: str) -> Optional[float]:
        try:
            obj = json.loads(text.strip())
            return float(obj["y"])
        except Exception:
            return None

    # ───────────────────────────── acquisition core ───────────────────────

    def _sample_candidates(self, t: int) -> List[np.ndarray]:
        """LLM-as-generative-sampler: propose a diverse candidate pool."""
        prompt = self._build_sampler_prompt(self.n_candidates)
        response = self._call_llm(t, prompt, max_tokens=64 * self.n_candidates)
        pts = self._parse_points(response["text"], self.n_candidates)
        if not pts:
            # Retry once before any fallback so the initial point stays
            # purely LLAMBO-driven (no rng).
            response = self._call_llm(t, prompt, max_tokens=64 * self.n_candidates)
            pts = self._parse_points(response["text"], self.n_candidates)
        if not pts:
            # Last-resort fallback: deterministic grid midpoint(s), not rng,
            # so different methods still diverge from a reproducible LLM start.
            lo = np.array([b[0] for b in self.task.bounds])
            hi = np.array([b[1] for b in self.task.bounds])
            mid = (lo + hi) / 2.0
            pts = [mid + (i / max(self.n_candidates, 1)) * (hi - mid)
                for i in range(self.n_candidates)]
        return pts

    def _score_candidates(self, t: int, candidates: List[np.ndarray]):
        """
        LLM-as-discriminative-surrogate: n_gens repeated predictions per
        candidate, mean/std used as the predictive posterior -- the same
        role a GP's (mu, sigma) plays in standard BO-EI.
        """
        means, stds = [], []
        for c in candidates:
            prompt = self._build_surrogate_prompt(c)
            preds = []
            for _ in range(self.n_gens):
                response = self._call_llm(t, prompt, max_tokens=32)
                val = self._parse_scalar(response["text"])
                if val is not None:
                    preds.append(val)
            if preds:
                means.append(float(np.mean(preds)))
                stds.append(max(float(np.std(preds)), 1e-6))
            else:
                means.append(0.0)
                stds.append(1e-6)
        return np.array(means), np.array(stds)

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        f_best = max(self._y)
        sigma = np.maximum(sigma, 1e-9)
        Z = (mu - f_best) / sigma
        return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def _sample_candidates_online(self, t: int, history, context: str) -> List[np.ndarray]:
        prompt = self._build_online_sampler_prompt(self.n_candidates, history, context)
        response = self._call_llm(t, prompt, max_tokens=64 * self.n_candidates)
        pts = self._parse_points(response["text"], self.n_candidates)
        if not pts:
            response = self._call_llm(t, prompt, max_tokens=64 * self.n_candidates)
            pts = self._parse_points(response["text"], self.n_candidates)
        if not pts:
            lo = np.array([b[0] for b in self.task.bounds])
            hi = np.array([b[1] for b in self.task.bounds])
            mid = (lo + hi) / 2.0
            pts = [mid + (i / max(self.n_candidates, 1)) * (hi - mid)
                   for i in range(self.n_candidates)]
        return pts
 
    def _score_candidates_online(self, t: int, candidates: List[np.ndarray], history, context: str):
        means, stds = [], []
        for c in candidates:
            prompt = self._build_online_surrogate_prompt(c, history, context)
            preds = []
            for _ in range(self.n_gens):
                response = self._call_llm(t, prompt, max_tokens=32)
                val = self._parse_scalar(response["text"])
                if val is not None:
                    preds.append(val)
            if preds:
                means.append(float(np.mean(preds)))
                stds.append(max(float(np.std(preds)), 1e-6))
            else:
                means.append(0.0)
                stds.append(1e-6)
        return np.array(means), np.array(stds)
 
    @staticmethod
    def _expected_improvement_from(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
        sigma = np.maximum(sigma, 1e-9)
        Z = (mu - f_best) / sigma
        return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)


    # ───────────────────────────── public API ──────────────────────────────

    def suggest(self, t: int) -> np.ndarray:
        candidates = self._sample_candidates(t)

        if not self._y:
            # Zero history: nothing to score candidates against yet (no
            # few-shot grounding for the surrogate, no f_best for EI).
            # Take the sampler's first LLM-generated proposal directly --
            # still fully LLAMBO-driven, just without surrogate ranking.
            return candidates[0]

        mu, sigma = self._score_candidates(t, candidates)
        ei = self._expected_improvement(mu, sigma)
        best_idx = int(np.argmax(ei))
        return candidates[best_idx]

    def observe(self, x: np.ndarray, y: float) -> None:
        self._history.append({"x": x, "y": y})
        self._X.append(x)
        self._y.append(y)

    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        candidates = self._sample_candidates_online(t, history, context)
 
        ys = [h["y"] for h in history]
        if not ys:
            # No history to ground the surrogate / f_best: take sampler's first.
            return candidates[0]
 
        mu, sigma = self._score_candidates_online(t, candidates, history, context)
        ei = self._expected_improvement_from(mu, sigma, max(ys))
        best_idx = int(np.argmax(ei))
        return candidates[best_idx]

    def step_cost(self, t: int) -> float:
        # return self._last_cost
        return self._tracker.cost_at(t)
    
    @property
    def cost_tracker(self) -> CostTracker:
        return self._tracker