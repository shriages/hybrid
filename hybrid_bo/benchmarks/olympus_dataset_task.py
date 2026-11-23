import hybrid_bo._matplotlib_compat  # noqa: F401  (must precede olympus import)


from olympus.datasets import list_datasets
print("DATASETS:", list_datasets())

import numpy as np
# Try loading a pretrained emulator — the critical test
from olympus import Emulator

import os
# hybrid_bo/benchmarks/olympus_dataset_task.py
import numpy as np
from hybrid_bo.benchmarks.base_task import BaseTask
# just as a check! then delete it!
for k in ['snar', 'suzuki', 'benzylation', 'alkox','colors_bob','colors_n9','fullerenes','hplc','photo_pce10','photo_wf3']:
    try:
        e = Emulator(dataset=k, model='BayesNeuralNet')
        print(f"{k:14s} dim={len(e.param_space)} goal={e.goal}  OK")
    except Exception as ex:
        print(f"{k:14s} FAILED: {type(ex).__name__}: {ex}")

    
    
class OlympusDatasetTask(BaseTask):
    """
    Olympus real-experiment dataset, evaluated through its pretrained
    emulator (a BayesNeuralNet surrogate trained on measured experimental
    data). These are REAL response surfaces (chemistry/materials), used here
    purely as realistic objective landscapes — the optimizer receives only
    abstract dimension/bounds, NO task identity or parameter semantics,
    consistent with the abstract-prompt design used for every other task.

    Objective sign: Olympus datasets define a 'goal' (minimize/maximize).
    We negate minimize-goal objectives so higher = better throughout.

    Subclasses fix `dataset_kind`.
    """
    dataset_kind: str = None
    model: str = "BayesNeuralNet"

    def __init__(self):
        from olympus import Emulator
        self._emu = Emulator(dataset=self.dataset_kind, model=self.model)

        ps = self._emu.param_space
        self.dim = len(ps)
        self.bounds = [(float(p.low), float(p.high)) for p in ps]

        # goal: 'minimize' -> negate so higher=better; 'maximize' -> as-is.
        goal = str(getattr(self._emu, "goal", "minimize")).lower()
        self._sign = -1.0 if goal == "minimize" else 1.0

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        x = np.clip(x, lo, hi)                       # emulator warns/errs out of bounds
        out = self._emu.run(x)                       # returns e.g. [[value]]
        val = float(np.asarray(out).ravel()[0])
        return self._sign * val                      # higher = better

    def f_star(self) -> float:
        # No closed-form optimum for an emulated real surface. Use best-found
        # fallback (None -> figure code uses best-across-methods), like Bayesmark.
        return None


class SnarTask(OlympusDatasetTask):
    name = "olympus_snar"
    dataset_kind = "snar"


class SuzukiTask(OlympusDatasetTask):
    name = "olympus_suzuki"
    dataset_kind = "suzuki"


class BenzylationTask(OlympusDatasetTask):
    name = "olympus_benzylation"
    dataset_kind = "benzylation"


class AlkoxTask(OlympusDatasetTask):
    name = "olympus_alkox"
    dataset_kind = "alkox"
    
class ColorsBobTask(OlympusDatasetTask):
    name = "olympus_colors_bob"
    dataset_kind = "colors_bob"


class ColorsN9Task(OlympusDatasetTask):
    name = "olympus_colors_n9"
    dataset_kind = "colors_n9"


class FullerenesTask(OlympusDatasetTask):
    name = "olympus_fullerenes"
    dataset_kind = "fullerenes"


class HplcTask(OlympusDatasetTask):
    name = "olympus_hplc"
    dataset_kind = "hplc"


class PhotoPce10Task(OlympusDatasetTask):
    name = "olympus_photo_pce10"
    dataset_kind = "photo_pce10"


class PhotoWf3Task(OlympusDatasetTask):
    name = "olympus_photo_wf3"
    dataset_kind = "photo_wf3"