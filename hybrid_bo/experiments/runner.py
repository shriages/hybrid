from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from hybrid_bo.experiments.config import ExperimentConfig
from hybrid_bo.core.run_result import RunResult


# ── SweepResult ───────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """
    Results from a full epsilon sweep over one (task, optimizer) pair.

    Attributes
    ----------
    config      : the base ExperimentConfig used (epsilon varied per entry)
    epsilons    : the ε values swept
    results     : results[i] = List[RunResult] across seeds for epsilons[i]
    """
    config:   ExperimentConfig
    epsilons: List[float]
    results:  List[List[RunResult]]   # shape: [n_epsilons, n_seeds]


# ── ExperimentRunner ──────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Builds optimizers from ExperimentConfig and executes them.
    Does not know about storage — call ResultStore separately.
    """

    # ── Task factory ──────────────────────────────────────────────────────────


    @staticmethod
    def _make_task(cfg: ExperimentConfig):
        """
        Builds the task instance from the full config — not just task_name —
        because ml_task needs ml_model_name and ml_dataset_name from cfg.
        """
        if cfg.task_name == "branin":
            from hybrid_bo.benchmarks.branin import BraninTask
            return BraninTask()

        if cfg.task_name == "hartmann6":
            from hybrid_bo.benchmarks.hartmann6 import Hartmann6Task
            return Hartmann6Task()
        
        if cfg.task_name.startswith("bayesmark_"):
            from hybrid_bo.benchmarks.bayesmark_task import BayesmarkTask
            parts      = cfg.task_name.split("_", 2)   # ["bayesmark", model, dataset]
            model_name = parts[1]
            dataset    = parts[2]
            return BayesmarkTask(model_name=model_name, dataset=dataset)

        if cfg.task_name.startswith("yahpo_"):
            from hybrid_bo.benchmarks.yahpo_task import YAHPOTask
            parts = cfg.task_name.split("_", 2)
            return YAHPOTask(scenario=parts[1], instance=parts[2])
        
        if cfg.task_name == "rastrigin":
            from hybrid_bo.benchmarks.rastrigin import RastriginTask
            return RastriginTask()

        if cfg.task_name == "schwefel":
            from hybrid_bo.benchmarks.schwefel import SchwefelTask
            return SchwefelTask()

        if cfg.task_name == "gallagher":
            from hybrid_bo.benchmarks.gallagher import GallagherTask
            return GallagherTask()

        if cfg.task_name == "lasso_diabetes":       # (online: task_name == ...)
            from hybrid_bo.benchmarks.lasso_bench_task import LassoDiabetesTask
            return LassoDiabetesTask()
        if cfg.task_name == "lasso_breast_cancer":
            from hybrid_bo.benchmarks.lasso_bench_task import LassoBreastCancerTask
            return LassoBreastCancerTask()
        
        if cfg.task_name.startswith("olympus_"):
                    from hybrid_bo.benchmarks.olympus_dataset_task import (
                        SnarTask, SuzukiTask, BenzylationTask, AlkoxTask,
                        ColorsBobTask, ColorsN9Task, FullerenesTask, HplcTask,
                        PhotoPce10Task, PhotoWf3Task,
                    )
                    _OLYMPUS = {
                        "olympus_snar": SnarTask,
                        "olympus_suzuki": SuzukiTask,
                        "olympus_benzylation": BenzylationTask,
                        "olympus_alkox": AlkoxTask,
                        "olympus_colors_bob": ColorsBobTask,
                        "olympus_colors_n9": ColorsN9Task,
                        "olympus_fullerenes": FullerenesTask,
                        "olympus_hplc": HplcTask,
                        "olympus_photo_pce10": PhotoPce10Task,
                        "olympus_photo_wf3": PhotoWf3Task,
                    }
                    if cfg.task_name in _OLYMPUS:
                        return _OLYMPUS[cfg.task_name]()
                    raise NotImplementedError(f"Unknown olympus task '{cfg.task_name}'.")

        raise NotImplementedError(
            f"Task '{cfg.task_name}' not yet implemented. "
            f"Add it here when ready."
        )

    # ── LLM factory ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_llm(model: str):
        import os
        if model.startswith("gpt"):
            from hybrid_bo.llm_backends.openai_llm import OpenAILLM
            return OpenAILLM(model=model)
        if model.startswith("claude"):
            from hybrid_bo.llm_backends.claude_llm import ClaudeLLM
            return ClaudeLLM(model=model)
        if model.startswith("deepseek"):
            from hybrid_bo.llm_backends.deepseek_llm import DeepSeekLLM
            return DeepSeekLLM(model=model)
        if model == "llama-local":
            from hybrid_bo.llm_backends.llama_llm import LlamaLLM
            return LlamaLLM()
        raise ValueError(f"Unknown LLM model '{model}'.")

    # ── Policy factory ────────────────────────────────────────────────────────

    @staticmethod
    def _make_policy(cfg: ExperimentConfig):
        if cfg.tau is not None:
            from hybrid_bo.switching.fixed_tau_policy import FixedTauPolicy
            return FixedTauPolicy(tau=cfg.tau)
        from hybrid_bo.switching.epsilon_policy import ProportionalEpsilonPolicy
        return ProportionalEpsilonPolicy(
            epsilon     = cfg.epsilon,
            metric      = cfg.metric,
            window      = cfg.window,
            warm_steps  = cfg.min_steps,
            min_steps   = cfg.min_steps,
            x_stars      = cfg.x_stars,
            aggregation = cfg.aggregation,
        )

    # ── Optimizer factory ─────────────────────────────────────────────────────

    @staticmethod
    def _make_optimizer(cfg: ExperimentConfig, task, seed: int):
        from hybrid_bo.core.gp_optimizer import GPOptimizer

        name = cfg.optimizer_name

        if name == "gp":
            return GPOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)

        if name == "tpe":                                    # <-- ADD (before _make_llm)
            from hybrid_bo.core.tpe_optimizer import TPEOptimizer
            return TPEOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
        
        if name == "smac":
            from hybrid_bo.core.smac_optimizer import SMACOptimizer
            return SMACOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
        
        llm = ExperimentRunner._make_llm(cfg.llm_model)

        if name == "llm":
            from hybrid_bo.core.llm_optimizer import LLMOptimizer
            return LLMOptimizer(task, T=cfg.T, llm=llm, seed=seed)

        if name == "llambo":
            from hybrid_bo.core.llambo_optimizer import LLAMBOOptimizer
            return LLAMBOOptimizer(
                task, T=cfg.T, llm=llm,
                n_init=cfg.n_init,
                n_candidates=cfg.n_candidates,
                seed=seed,
            )

        if name == "lgbo":
            from hybrid_bo.core.lgbo_optimizer import LGBOOptimizer
            return LGBOOptimizer(
                task, T=cfg.T, llm=llm,
                n_init=cfg.n_init,
                seed=seed,
            )

        # Hybrid variants
        if name in {"hybrid_llm", "hybrid_llambo", "hybrid_lgbo"}:
            from hybrid_bo.core.hybrid_optimizer import HybridOptimizer

            gp_opt = GPOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
            policy = ExperimentRunner._make_policy(cfg)

            if name == "hybrid_llm":
                from hybrid_bo.core.llm_optimizer import LLMOptimizer
                llm_opt = LLMOptimizer(task, T=cfg.T, llm=llm, seed=seed)
            elif name == "hybrid_llambo":
                from hybrid_bo.core.llambo_optimizer import LLAMBOOptimizer
                llm_opt = LLAMBOOptimizer(
                    task, T=cfg.T, llm=llm,
                    n_init=cfg.n_init,
                    n_candidates=cfg.n_candidates,
                    seed=seed,
                )
            else:
                from hybrid_bo.core.lgbo_optimizer import LGBOOptimizer
                llm_opt = LGBOOptimizer(
                    task, T=cfg.T, llm=llm,
                    n_init=cfg.n_init,
                    seed=seed,
                )
            # policy is only with epsilon
            return HybridOptimizer(llm_opt, gp_opt, policy, task, cfg.T)

        if name.startswith("fixed_switch_"):
            from hybrid_bo.core.fixed_switch_optimizer import FixedSwitchOptimizer
            from hybrid_bo.core.llm_optimizer import LLMOptimizer
            from hybrid_bo.core.llambo_optimizer import LLAMBOOptimizer
            from hybrid_bo.core.lgbo_optimizer import LGBOOptimizer

            arm = name.split("fixed_switch_", 1)[1]          # llm | llambo | lgbo
            if arm == "llm":
                llm_opt = LLMOptimizer(task, T=cfg.T, llm=llm, seed=seed)
            elif arm == "llambo":
                llm_opt = LLAMBOOptimizer(
                    task, T=cfg.T, llm=llm,
                    n_init=cfg.n_init, n_candidates=cfg.n_candidates, seed=seed,
                )
            elif arm == "lgbo":
                llm_opt = LGBOOptimizer(task, T=cfg.T, llm=llm, n_init=cfg.n_init, seed=seed)
            else:
                raise ValueError(f"Unknown fixed_switch arm '{arm}'.")

            gp_opt = GPOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
            return FixedSwitchOptimizer(
                task=task, llm_opt=llm_opt, gp_opt=gp_opt,
                T=cfg.T, tau=cfg.T // 2, llm_name=arm, seed=seed,
            )

        raise ValueError(f"Unknown optimizer '{name}'.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_single(self, cfg: ExperimentConfig, seed: int) -> RunResult:
        """Run one optimizer for one seed. Returns a RunResult."""
        # task = self._make_task(cfg.task_name)
        task = self._make_task(cfg)  
        opt  = self._make_optimizer(cfg, task, seed)  # if task contain string "hybrid", then also make policy! 
        # opt is an optimizer, have method run(), which returns a RunResult with tau maybe changed!
        return opt.run() 

    def run_seeds(self, cfg: ExperimentConfig) -> List[RunResult]:
        """Run one optimizer over all seeds in cfg.seeds."""
        return [self.run_single(cfg, s) for s in cfg.seeds]

    def run_epsilon_sweep(
        self,
        cfg: ExperimentConfig,
        epsilons: List[float],
    ) -> SweepResult:
        """
        Sweep over epsilon values. For each ε, run all seeds.
        cfg.epsilon is overridden per iteration; cfg.tau must be None.

        Returns a SweepResult containing all RunResults.
        """
        from dataclasses import replace

        all_results = []
        for eps in epsilons:
            cfg_eps = replace(cfg, epsilon=eps, tau=None)
            seed_results = self.run_seeds(cfg_eps)
            all_results.append(seed_results)

        return SweepResult(
            config=cfg,
            epsilons=epsilons,
            results=all_results,
        )

    def run_all_baselines(self, task_name: str, T: int,
                          seeds: List[int]) -> dict:
        """
        Run the four non-switching baselines (gp, llm, llambo, lgbo)
        for a given task. Returns dict keyed by optimizer name.

        Note: llm/llambo/lgbo baselines need a real LLM — set API keys first.
        For offline testing, run only 'gp'.
        """
        baselines = {}
        for opt_name in ["gp", "llm", "llambo", "lgbo"]:
            cfg = ExperimentConfig(
                task_name=task_name,
                optimizer_name=opt_name,
                T=T,
                seeds=seeds,
                tau=T,     # tau=T means never switches — full LLM run
            )
            baselines[opt_name] = self.run_seeds(cfg)
        return baselines