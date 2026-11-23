"""
LGBOOptimizer: faithful implementation of LLM-Guided Bayesian Optimization
(LGBO), following "Unleashing LLMs in Bayesian Optimization: Preference-Guided
Framework for Scientific Discovery" (arXiv:2605.17976).

Core idea from the paper (Sec. 3.2 "Region-Lifted Preference" + Sec. 3.3
"LGBO Framework"):

  1. At every iteration the LLM is asked for a coarse preference, constrained
     to one of two output formats:
         point mode:  ["point",  [x1, ..., xd], c]
         region mode: ["region", [[lb1,...,lbd], [ub1,...,ubd]], c]
     where c in [0, 1] is a confidence score.

  2. The preferred region/point is discretized into a small grid
     G = {x_g}_{g=1}^{G} with non-negative weights a_g (uniform 1/G for a
     region; a single grid point with weight 1 for a point preference -- the
     kernel itself supplies the "smoothly decaying" neighborhood weighting).

  3. Proposition 1 in the paper shows that exponentially tilting the GP
     posterior by exp(lambda * a^T F_G) is EXACTLY equivalent to leaving the
     covariance untouched and shifting the mean:

         mu_lambda(x) = mu(x) + lambda * sum_g a_g * k(x, x_g)

     i.e. for any finite query set X (our candidate pool):

         F_X ~ N( mu_X + lambda * Sigma_{X,G} @ a ,  Sigma_{X,X} )

  4. The guidance strength lambda is calibrated from the LLM's confidence c
     via

         lambda = c / sqrt(a^T Sigma_{G,G} a)

     so low-uncertainty regions get a small effective shift and
     high-uncertainty regions get a larger one, with no extra hyperparameter.

  5. The acquisition function (here: Expected Improvement) is then evaluated
     on the mean-shifted, covariance-preserving surrogate. This mean shift --
     not a filtered/clipped candidate pool, and not an override of the
     acquisition function -- is the actual LGBO mechanism.

  6. Before the GP has enough data (t < n_init), LGBO uses the LLM purely as
     a warm-start proposer (Sec 3.3, "From warm-start to continuous
     guidance"): the LLM's point/region suggestion is used directly to pick
     the next query, since there is no surrogate yet to lift.

This file merges the former `LGBOOptimizer` and `LGBOPromptBuilder` into a
single class. All external touchpoints used elsewhere in the codebase are
preserved: `LGBOOptimizer(task, T, llm, n_init=..., seed=...)`, `.name`,
`.suggest(t)`, `.observe(x, y)`, `.step_cost(t)`, `.cost_tracker`.
"""

import json
import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# from hybrid_bo.utils.extract_json import _extract_json
from .base_optimizer import BaseOptimizer
from hybrid_bo.llm_backends.cost_tracker import CostTracker
from hybrid_bo.benchmarks.task_metadata import describe_task



def _strip_code_fence(text: str) -> str:
    """Return the JSON payload from an LLM reply.

    Models differ in how they wrap structured output: DeepSeek and GPT
    typically emit bare JSON, while Claude wraps it in a markdown fence
    (```json ... ```) by default. Some models also add a sentence of prose.
    Normalising here keeps every backend on the same parse path.
    """
    s = (text or "").strip()

    # ```json ... ```   or   ``` ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()

    # still not starting at a JSON token? take the first {...} or [...] span
    if s and s[0] not in "{[":
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            s = m.group(1)

    return s.strip()


class LGBOOptimizer(BaseOptimizer):
    """
    LGBO optimizer implementing the region-lifted preference mechanism.

    At every step (after warm-start) the LLM supplies a point or region
    preference with a confidence score. This is converted into a mean shift
    of the GP surrogate (covariance untouched), and EI is then computed on
    the shifted surrogate to pick the next query -- matching the paper's
    "region-lifted preference" formalism exactly, rather than clipping/
    filtering the candidate pool.
    """

    name = "lgbo"

    # Number of Sobol points used to discretize an LLM-preferred region.
    _GRID_SIZE = 32
    # Numerical floor to avoid division by ~0 when computing lambda.
    _VAR_EPS = 1e-9

    def __init__(self, task, T: int, llm,
                 n_init: int = 1, seed: int = 0):
        super().__init__(task, T, seed)
        self._llm = llm
        self._tracker = CostTracker()
        self._history = []
        self._last_cost = 0.0
        self.n_init = n_init

        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        )
        self._X, self._y = [], []

        self._lower = np.array([b[0] for b in self.task.bounds], dtype=float)
        self._upper = np.array([b[1] for b in self.task.bounds], dtype=float)

        # Most recent parsed LLM preference: {"mode", "point"/"region", "c"}
        self._preference = None

    # ------------------------------------------------------------------ #
    # Prompt construction (formerly LGBOPromptBuilder)
    # ------------------------------------------------------------------ #
    def _task_description(self) -> str:
        # Task-specific semantic description resolved from the task's `.name`
        # (objective/direction, per-variable meanings, real ranges/scales, and
        # real-experiment background). This grounds the LLM's point/region
        # preference in domain knowledge, in the spirit of the LGBO prompt's
        # "evidence hierarchy" (background over raw history). The point/region
        # confidence JSON contract and the raw-coordinate parsing are unchanged;
        # only the descriptive wording is enriched.
        return describe_task(self.task)

    @staticmethod
    def _format_history(history, max_entries: int = 50) -> str:
        trimmed = history[-max_entries:]
        if not trimmed:
            return "(no observations yet)"
        lines = []
        for i, entry in enumerate(trimmed):
            x_str = ", ".join(f"{v:.4f}" for v in np.asarray(entry["x"]).ravel())
            lines.append(f"  step {i}: x=[{x_str}]  y={entry['y']:.4f}")
        return "\n".join(lines)

    def _build_prompt(self) -> str:
        history_str = self._format_history(self._history, max_entries=50)
        return (
            f"You are a domain expert guiding a Bayesian optimizer.\n\n"
            f"Task:\n{self._task_description()}\n\n"
            f"Full optimization trajectory so far:\n{history_str}\n\n"
            f"Based on observed patterns and domain knowledge, provide a "
            f"coarse preference for where to search next. You MUST respond "
            f"with exactly one of the following two JSON array formats, and "
            f"nothing else:\n\n"
            f"  Point mode:  [\"point\", [x1, x2, ..., xd], c]\n"
            f"  Region mode: [\"region\", [[lb1, ..., lbd], [ub1, ..., ubd]], c]\n\n"
            f"where d = {self.task.dim} is the input dimension and c is a "
            f"confidence score in [0, 1] reflecting how strongly you believe "
            f"this point/region is promising. Return ONLY the JSON array."
        )


    def online_prompt_builder(self, history, context: str = "") -> str:
        history_str = self._format_history(history, max_entries=50)
        ctx = f"{context}\n\n" if context else ""
        return (
            f"{ctx}"
            f"You are a domain expert guiding a Bayesian optimizer.\n\n"
            f"Task:\n{self._task_description()}\n\n"
            f"Full optimization trajectory so far:\n{history_str}\n\n"
            f"Based on observed patterns and domain knowledge, provide a "
            f"coarse preference for where to search next. You MUST respond "
            f"with exactly one of the following two JSON array formats, and "
            f"nothing else:\n\n"
            f"  Point mode:  [\"point\", [x1, x2, ..., xd], c]\n"
            f"  Region mode: [\"region\", [[lb1, ..., lbd], [ub1, ..., ubd]], c]\n\n"
            f"where d = {self.task.dim} is the input dimension and c is a "
            f"confidence score in [0, 1] reflecting how strongly you believe "
            f"this point/region is promising. Return ONLY the JSON array."
        )

    # ------------------------------------------------------------------ #
    # LLM querying + parsing
    # ------------------------------------------------------------------ #
    def _query_llm(self, t: int, max_tokens: int = 256, json_only=True) -> dict:
        prompt = self._build_prompt()
        response = self._llm.complete(prompt, max_tokens=max_tokens)
        self._last_cost = response["cost"]
        self._tracker.record(
            t=t, model=self._llm.model, optimizer=self.name,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=response["cost"],
        )
        pref = self._parse_preference(response["text"])
        resp_text = response["text"]
        print(f"[{self.name} t={t}] raw={resp_text!r}")

        if pref is not None:
            self._preference = pref
        return self._preference

    def _query_llm_online(self, t: int, history, context: str, max_tokens: int = 256):
        prompt = self.online_prompt_builder(history, context)
        response = self._llm.complete(prompt, max_tokens=max_tokens)
        self._last_cost = response["cost"]
        self._tracker.record(
            t=t, model=self._llm.model, optimizer=self.name,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=response["cost"],
        )
        resp_text = response["text"]
        print(f"[{self.name} online t={t}] raw={resp_text!r}")
        return self._parse_preference(response["text"])


    def _parse_preference(self, text: str):
        """Parse the ["point"|"region", payload, confidence] LLM output."""
        try:
            obj = json.loads(_strip_code_fence(text))
            if not isinstance(obj, list) or len(obj) != 3:
                return None
            mode, payload, c = obj
            c = float(np.clip(float(c), 0.0, 1.0))

            if mode == "point":
                point = np.asarray(payload, dtype=float)
                if point.shape != (self.task.dim,):
                    return None
                return {"mode": "point", "point": point, "c": c}

            if mode == "region":
                x_min = np.asarray(payload[0], dtype=float)
                x_max = np.asarray(payload[1], dtype=float)
                if x_min.shape != (self.task.dim,) or x_max.shape != (self.task.dim,):
                    return None
                return {"mode": "region", "x_min": x_min, "x_max": x_max, "c": c}

        except Exception:
            pass
        return None


    # ------------------------------------------------------------------ #
    # Region-lifted preference machinery (Sec. 3.2 of the paper)
    # ------------------------------------------------------------------ #
    def _grid_and_weights(self, preference: dict):
        """
        Build the discretization grid G = {x_g} and non-negative weights a_g
        for the LLM's point/region preference.
        """
        if preference["mode"] == "point":
            point = np.clip(preference["point"], self._lower, self._upper)
            grid = point.reshape(1, -1)
            weights = np.array([1.0])
            return grid, weights

        # region mode: clip to task bounds, fall back to full bounds if
        # the LLM gave a degenerate/invalid rectangle.
        lb = np.maximum(preference["x_min"], self._lower)
        ub = np.minimum(preference["x_max"], self._upper)
        if np.any(lb >= ub):
            lb, ub = self._lower, self._upper

        sobol_seed = int(self.rng.integers(0, 1_000_000))
        sampler = qmc.Sobol(d=self.task.dim, scramble=True, seed=sobol_seed)
        unit_grid = sampler.random(self._GRID_SIZE)
        grid = qmc.scale(unit_grid, lb, ub)
        weights = np.full(self._GRID_SIZE, 1.0 / self._GRID_SIZE)
        return grid, weights

    def _lifted_ei(self, pool: np.ndarray, preference: dict) -> np.ndarray:
        """
        Compute Expected Improvement on the region-lifted surrogate:
        mean shifted by lambda * Sigma_{X,G} @ a, covariance unchanged.
        Falls back to plain EI (lambda effectively 0) if the region has
        ~zero posterior uncertainty or preference is missing.
        """
        f_best = max(self._y)

        if preference is None:
            mu, sigma = self._gp.predict(pool, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - f_best) / sigma
            return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

        grid, a = self._grid_and_weights(preference)
        joint_X = np.vstack([pool, grid])
        n_pool = pool.shape[0]

        mu_joint, cov_joint = self._gp.predict(joint_X, return_cov=True)
        mu_pool = mu_joint[:n_pool]
        Sigma_XX_diag = np.clip(np.diag(cov_joint)[:n_pool], 1e-12, None)
        Sigma_XG = cov_joint[:n_pool, n_pool:]
        Sigma_GG = cov_joint[n_pool:, n_pool:]

        quad = float(a @ Sigma_GG @ a)
        quad = max(quad, self._VAR_EPS)
        lam = preference["c"] / np.sqrt(quad)

        mu_lifted = mu_pool + lam * (Sigma_XG @ a)
        sigma = np.sqrt(Sigma_XX_diag)

        Z = (mu_lifted - f_best) / sigma
        return (mu_lifted - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def _lifted_ei_with(self, gp, ys, pool: np.ndarray, preference: dict) -> np.ndarray:
        f_best = max(ys)
 
        if preference is None:
            mu, sigma = gp.predict(pool, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - f_best) / sigma
            return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
 
        grid, a = self._grid_and_weights(preference)
        joint_X = np.vstack([pool, grid])
        n_pool = pool.shape[0]
 
        mu_joint, cov_joint = gp.predict(joint_X, return_cov=True)
        mu_pool = mu_joint[:n_pool]
        Sigma_XX_diag = np.clip(np.diag(cov_joint)[:n_pool], 1e-12, None)
        Sigma_XG = cov_joint[:n_pool, n_pool:]
        Sigma_GG = cov_joint[n_pool:, n_pool:]
 
        quad = float(a @ Sigma_GG @ a)
        quad = max(quad, self._VAR_EPS)
        lam = preference["c"] / np.sqrt(quad)
 
        mu_lifted = mu_pool + lam * (Sigma_XG @ a)
        sigma = np.sqrt(Sigma_XX_diag)
 
        Z = (mu_lifted - f_best) / sigma
        return (mu_lifted - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)


    # ------------------------------------------------------------------ #
    # BaseOptimizer interface
    # ------------------------------------------------------------------ #
    def suggest(self, t: int) -> np.ndarray:
        # ── Warm-start phase: no GP data yet -> LLM proposes directly ──
        if len(self._X) < self.n_init:
            preference = self._query_llm(t)
            if preference is None:
                # Retry once; the first point must be LLM-proposed, never rng.
                preference = self._query_llm(t)
            if preference is None:
                raise RuntimeError(
                    "LGBO warm-start requires a valid LLM preference but "
                    "parsing failed twice; check the LLM output format."
                )
            if preference["mode"] == "point":
                return np.clip(preference["point"], self._lower, self._upper)
            # region mode: use the region center (deterministic, no rng)
            lb = np.maximum(preference["x_min"], self._lower)
            ub = np.minimum(preference["x_max"], self._upper)
            if np.any(lb >= ub):
                lb, ub = self._lower, self._upper
            return 0.5 * (lb + ub)

        # ── Main phase: LLM injects a region-lifted preference, EI picks ──
        # ── the next point on the mean-shifted (cov-preserved) surrogate ──
        preference = self._query_llm(t)

        pool = self.rng.uniform(self._lower, self._upper, size=(500, self.task.dim))
        ei = self._lifted_ei(pool, preference)
        return pool[np.argmax(ei)]

    def observe(self, x: np.ndarray, y: float) -> None:
        self._history.append({"x": x, "y": y})
        self._X.append(x)
        self._y.append(y)
        if len(self._X) >= self.n_init:
            self._gp.fit(np.array(self._X), np.array(self._y))
            
    def suggest_online(self, t: int, history: list, context: str = "") -> np.ndarray:
        # Warm-start: not enough global data to fit the GP -> LLM proposes.
        if len(history) < self.n_init:
            preference = self._query_llm_online(t, history, context)
            if preference is None:
                preference = self._query_llm_online(t, history, context)
            if preference is None:
                raise RuntimeError(
                    "LGBO online warm-start requires a valid LLM preference "
                    "but parsing failed twice; check the LLM output format."
                )
            if preference["mode"] == "point":
                return np.clip(preference["point"], self._lower, self._upper)
            lb = np.maximum(preference["x_min"], self._lower)
            ub = np.minimum(preference["x_max"], self._upper)
            if np.any(lb >= ub):
                lb, ub = self._lower, self._upper
            return 0.5 * (lb + ub)
 
        # Main phase: fit a LOCAL GP on the GLOBAL history, then lifted-EI.
        X = np.array([np.asarray(h["x"], dtype=float) for h in history])
        y = [float(h["y"]) for h in history]
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.seed,
        )
        gp.fit(X, np.array(y))
 
        preference = self._query_llm_online(t, history, context)
        pool = self.rng.uniform(self._lower, self._upper, size=(500, self.task.dim))
        ei = self._lifted_ei_with(gp, y, pool, preference)
        return pool[np.argmax(ei)]


    def step_cost(self, t: int) -> float:
        # return self._last_cost
        return self._tracker.cost_at(t)
    
    @property
    def cost_tracker(self) -> CostTracker:
        return self._tracker