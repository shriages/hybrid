from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
# hybrid_bo/benchmarks/base_task.py — ADD this, don't remove existing methods
from typing import Dict, Any, Optional
from dataclasses import dataclass


class BaseTask(ABC):
    """Abstract benchmark task. Convention: higher evaluate() = better."""

    dim: int
    bounds: List[Tuple[float, float]]   # [(lo, hi), ...] length == dim
    name: str

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Query the objective. x has shape (dim,). Returns scalar (higher = better)."""
        ...

    @abstractmethod
    def f_star(self) -> float:
        """Ground truth optimum value (negated if minimization problem)."""
        ...

    def random_x(self, rng: np.random.Generator) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi)

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return np.clip(x, lo, hi)
    
    # ---------ML bayes benchmarks specific -----------------
    # Still inside BaseTask — add these as optional overrides
    def hyperparam_specs(self) -> Optional[List["HyperparamSpec"]]:
        """
        Returns None for continuous tasks (Branin, Hartmann6).
        Returns a list of HyperparamSpec for mixed-type tasks (MLTask).
        Optimizers check this to know whether to treat x as raw floats
        or decode it through the spec list.
        """
        return None

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Maps a raw [0,1]^d vector (what GP/LLM optimizers operate on)
        into actual hyperparameter values (ints, categoricals, log-scaled floats).
        Default: identity, used by continuous tasks.
        """
        return {f"x{i}": v for i, v in enumerate(x)}





@dataclass
class HyperparamSpec:
    """
    Describes one hyperparameter's search space.
    type: "float" | "int" | "categorical"
    For float/int: low, high define the range.
    For categorical: choices is the list of valid values.
    log_scale: if True, sample/encode in log space (common for learning rates, C, alpha).
    """
    name:       str
    type:       str
    low:        Optional[float] = None
    high:       Optional[float] = None
    choices:    Optional[List[Any]] = None
    log_scale:  bool = False

