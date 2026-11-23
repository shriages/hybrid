"""
OnlineExperimentRunner: builds arm optimizers from an OnlineConfig and runs
the OnlineHybridOptimizer. Pure (no disk I/O) — saving lives in
scripts/run_online_local.py, mirroring ExperimentRunner + run_local.py.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List
import numpy as np

from hybrid_bo.experiments.runner import ExperimentRunner
from hybrid_bo.experiments.online_config import OnlineConfig
from hybrid_bo.core.online_hybrid_optimizer import OnlineHybridOptimizer
from hybrid_bo.core.online_run_result import OnlineRunResult


@dataclass
class OnlineLambdaSweep:
    """In-memory bundle of a λ-sweep. Not persisted as one object; the
    driver saves each (λ, seed) RunResult separately. Handy for tests/notebooks."""
    config: OnlineConfig
    lambdas: List[float]
    results: List[List[OnlineRunResult]]   # [n_lambdas][n_seeds]




class OnlineExperimentRunner:

    # ── Task factory (inlined from ExperimentRunner._make_task; reads only
    #    task_name, so no fragile coupling to ExperimentConfig internals) ─────
    @staticmethod
    def _make_task(task_name: str):
        if task_name == "branin":
            from hybrid_bo.benchmarks.branin import BraninTask
            return BraninTask()
        if task_name == "hartmann6":
            from hybrid_bo.benchmarks.hartmann6 import Hartmann6Task
            return Hartmann6Task()
        if task_name.startswith("bayesmark_"):
            from hybrid_bo.benchmarks.bayesmark_task import BayesmarkTask
            _, model_name, dataset = task_name.split("_", 2)
            return BayesmarkTask(model_name=model_name, dataset=dataset)
        if task_name.startswith("yahpo_"):
            from hybrid_bo.benchmarks.yahpo_task import YAHPOTask
            _, scenario, instance = task_name.split("_", 2)
            return YAHPOTask(scenario=scenario, instance=instance)
        if task_name == "rastrigin":
            from hybrid_bo.benchmarks.rastrigin import RastriginTask
            return RastriginTask()
        if task_name == "schwefel":
            from hybrid_bo.benchmarks.schwefel import SchwefelTask
            return SchwefelTask()
        if task_name == "gallagher":
            from hybrid_bo.benchmarks.gallagher import GallagherTask
            return GallagherTask()

        if task_name == "lasso_diabetes":       # (online: task_name == ...)
            from hybrid_bo.benchmarks.lasso_bench_task import LassoDiabetesTask
            return LassoDiabetesTask()
        if task_name == "lasso_breast_cancer":
            from hybrid_bo.benchmarks.lasso_bench_task import LassoBreastCancerTask
            return LassoBreastCancerTask()

        if task_name.startswith("olympus_"):
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
                    if task_name in _OLYMPUS:
                        return _OLYMPUS[task_name]()
                    raise NotImplementedError(f"Unknown olympus task '{task_name}'.")
        raise NotImplementedError(f"Task '{task_name}' not implemented.")

    # ── Arm factory: one arm per name, shared LLM backend ────────────────────
    @staticmethod
    def _make_arms(cfg: OnlineConfig, task, seed: int):
        from hybrid_bo.core.gp_optimizer import GPOptimizer
        from hybrid_bo.core.llm_optimizer import LLMOptimizer
        from hybrid_bo.core.llambo_optimizer import LLAMBOOptimizer
        from hybrid_bo.core.lgbo_optimizer import LGBOOptimizer

        needs_llm = any(a in {"llm", "llambo", "lgbo"} for a in cfg.arms)
        llm = ExperimentRunner._make_llm(cfg.llm_model) if needs_llm else None

        arms, names = [], []
        for a in cfg.arms:
            if a == "gp":
                arm = GPOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
            elif a == "tpe":
                from hybrid_bo.core.tpe_optimizer import TPEOptimizer
                arm = TPEOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
            elif a == "llm":
                arm = LLMOptimizer(task, T=cfg.T, llm=llm, seed=seed)
            elif a == "llambo":
                arm = LLAMBOOptimizer(
                    task, T=cfg.T, llm=llm,
                    n_init=cfg.n_init, n_candidates=cfg.n_candidates, seed=seed,
                )
            elif a == "lgbo":
                arm = LGBOOptimizer(task, T=cfg.T, llm=llm, n_init=cfg.n_init, seed=seed)
            else:
                raise ValueError(f"Unknown arm '{a}'.")
            arms.append(arm)
            names.append(a)
        return arms, names


    # ── Arm factory for the one-time-switch ablation ─────────────────────────
    #    Builds exactly TWO arms: the chosen LLM arm, and GP. Reuses the same
    #    constructors as _make_arms so the arms are identical to the ones the
    #    Exp3 runs use (no separate code path for arm construction).
    @staticmethod
    def _make_one_time_arms(cfg: OnlineConfig, task, seed: int):
        from hybrid_bo.core.gp_optimizer import GPOptimizer
        from hybrid_bo.core.llm_optimizer import LLMOptimizer
        from hybrid_bo.core.llambo_optimizer import LLAMBOOptimizer
        from hybrid_bo.core.lgbo_optimizer import LGBOOptimizer

        name = cfg.one_time_arm            # "llm" | "llambo" | "lgbo"
        llm = ExperimentRunner._make_llm(cfg.llm_model)

        if name == "llm":
            llm_arm = LLMOptimizer(task, T=cfg.T, llm=llm, seed=seed)
        elif name == "llambo":
            llm_arm = LLAMBOOptimizer(
                task, T=cfg.T, llm=llm,
                n_init=cfg.n_init, n_candidates=cfg.n_candidates, seed=seed,
            )
        elif name == "lgbo":
            llm_arm = LGBOOptimizer(task, T=cfg.T, llm=llm, n_init=cfg.n_init, seed=seed)
        else:
            raise ValueError(f"one_time arm must be llm|llambo|lgbo, got '{name}'.")

        gp_arm = GPOptimizer(task, T=cfg.T, n_init=cfg.n_init, seed=seed)
        return llm_arm, gp_arm, name

    # ── Runs ─────────────────────────────────────────────────────────────────
    def run_single(self, cfg: OnlineConfig, seed: int) -> OnlineRunResult:
        task = self._make_task(cfg.task_name)

        # Ablation: deterministic one-time switch (no bandit at all).
        if getattr(cfg, "switch_mode", None) is not None:
            from hybrid_bo.core.one_time_switch_optimizer import OneTimeSwitchOptimizer
            llm_arm, gp_arm, llm_name = self._make_one_time_arms(cfg, task, seed)
            opt = OneTimeSwitchOptimizer(
                task=task,
                llm_arm=llm_arm,
                gp_arm=gp_arm,
                llm_name=llm_name,
                T=cfg.T,
                tau=cfg.effective_tau,
                lambda_=cfg.lambda_,     # echoed for bookkeeping only
                seed=seed,
            )
            return opt.run()

        # Normal Exp3 online switching.
        arms, names = self._make_arms(cfg, task, seed)
        opt = OnlineHybridOptimizer(
            task=task, arms=arms, arm_names=names,
            T=cfg.T, b=cfg.b, b_first=cfg.b_first,
            eta=cfg.eta, lambda_=cfg.lambda_, alpha=cfg.alpha,
            eps1=cfg.eps1, eps2=cfg.eps2, seed=seed,
        )
        return opt.run()

    def run_seeds(self, cfg: OnlineConfig) -> List[OnlineRunResult]:
        return [self.run_single(cfg, s) for s in cfg.seeds]

    def run_lambda_sweep(self, cfg: OnlineConfig, lambdas: List[float]) -> OnlineLambdaSweep:
        all_results = [self.run_seeds(replace(cfg, lambda_=lam)) for lam in lambdas]
        return OnlineLambdaSweep(config=cfg, lambdas=lambdas, results=all_results)