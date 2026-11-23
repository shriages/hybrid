"""
Load saved online-switching results into plain lists/arrays for plotting.
You draw the figures yourself; this just reads the per-λ per-seed pickles.
"""
from __future__ import annotations
from typing import List
from hybrid_bo.experiments.result_store import ResultStore


def _opt_name(lambda_: float) -> str:
    return f"online_hybrid_lambda_{lambda_:.2f}"


def load_lambda_seed(store: ResultStore, task: str, model: str,
                     lambda_: float, seed: int):
    """Return one OnlineRunResult (all fields are plain lists / ndarrays)."""
    return store.load(task, _opt_name(lambda_), model, seed)


def load_lambda_all_seeds(store: ResultStore, task: str, model: str,
                          lambda_: float, seeds: List[int]):
    """List[OnlineRunResult] across seeds for one λ (skips missing)."""
    out = []
    for s in seeds:
        if store.exists(task, _opt_name(lambda_), model, s):
            out.append(store.load(task, _opt_name(lambda_), model, s))
    return out