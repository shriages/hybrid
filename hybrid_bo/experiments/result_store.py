import os
import pickle
from pathlib import Path
from typing import List, Optional

from hybrid_bo.core.run_result import RunResult
from hybrid_bo.experiments.runner import SweepResult
from hybrid_bo.experiments.config import ExperimentConfig

class ResultStore:
    """
    Saves and loads RunResults to/from disk.

    Directory layout under root_dir:
        raw/
            {task}/{optimizer}/{model}/seed_{seed}.pkl
        sweeps/
            {task}/{optimizer}/{model}/{metric}_sweep.pkl
    """

    def __init__(self, root_dir: str = "results"):
        self.root = Path(root_dir)
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        (self.root / "sweeps").mkdir(parents=True, exist_ok=True)
        # self.root = "results"
        
    # ── Key builders ──────────────────────────────────────────────────────────

    def _raw_path(
        self,
        task: str,
        optimizer: str,
        model: str,
        seed: int,
    ) -> Path:
        d = self.root / "raw" / task / optimizer / model
        d.mkdir(parents=True, exist_ok=True)
        print(d / f"seed_{seed}.pkl")
        return d / f"seed_{seed}.pkl"

    def _sweep_path(
        self,
        task: str,
        optimizer: str,
        model: str,
        metric: str,
    ) -> Path:
        d = self.root / "sweeps" / task / optimizer / model
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{metric}_sweep.pkl"

    # ── Single RunResult ──────────────────────────────────────────────────────

    def save(
        self,
        result: RunResult,
        cfg: ExperimentConfig,
        seed: int,
    ) -> Path:
        path = self._raw_path(
            cfg.task_name, cfg.optimizer_name, cfg.llm_model, seed
        )
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return path

    def load(
        self,
        task: str,
        optimizer: str,
        model: str,
        seed: int,
    ) -> RunResult:
        path = self._raw_path(task, optimizer, model, seed)
        if not path.exists():
            raise FileNotFoundError(f"No result at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def exists(
        self,
        task: str,
        optimizer: str,
        model: str,
        seed: int,
    ) -> bool:
        return self._raw_path(task, optimizer, model, seed).exists()

    # ── SweepResult ───────────────────────────────────────────────────────────

    def save_sweep(self, sweep: SweepResult) -> Path:
        cfg = sweep.config
        path = self._sweep_path(
            cfg.task_name, cfg.optimizer_name,
            cfg.llm_model, cfg.metric,
        )
        with open(path, "wb") as f:
            pickle.dump(sweep, f)
        return path

    def load_sweep(
        self,
        task: str,
        optimizer: str,
        model: str,
        metric: str,
    ) -> SweepResult:
        path = self._sweep_path(task, optimizer, model, metric)
        if not path.exists():
            raise FileNotFoundError(f"No sweep at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def sweep_exists(
        self,
        task: str,
        optimizer: str,
        model: str,
        metric: str,
    ) -> bool:
        return self._sweep_path(task, optimizer, model, metric).exists()

    # ── Bulk helpers ──────────────────────────────────────────────────────────

    def save_all_seeds(
        self,
        results: List[RunResult],
        cfg: ExperimentConfig,
    ) -> List[Path]:
        return [
            self.save(r, cfg, seed)
            for r, seed in zip(results, cfg.seeds)
        ]

    def load_all_seeds(
        self,
        task: str,
        optimizer: str,
        model: str,
        seeds: List[int],
    ) -> List[RunResult]:
        return [self.load(task, optimizer, model, s) for s in seeds]

    def list_available(self) -> dict:
        """Return a nested dict of all saved results."""
        available = {}
        raw_root = self.root / "raw"
        if not raw_root.exists():
            return available
        for task_dir in raw_root.iterdir():
            if not task_dir.is_dir():
                continue
            available[task_dir.name] = {}
            for opt_dir in task_dir.iterdir():
                if not opt_dir.is_dir():
                    continue
                available[task_dir.name][opt_dir.name] = {}
                for model_dir in opt_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    seeds = [
                        int(p.stem.split("_")[1])
                        for p in model_dir.glob("seed_*.pkl")
                    ]
                    available[task_dir.name][opt_dir.name][model_dir.name] = seeds
        return available