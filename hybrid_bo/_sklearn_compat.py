"""
Compatibility shim: sklearn >=1.2 removed load_boston, and accessing
it raises ImportError (not AttributeError) via sklearn's module-level
__getattr__. So hasattr() doesn't work here — we need try/except.
We never call load_boston directly (we don't use the 'boston' task),
so a lazy-raising stub is enough to satisfy bayesmark's import-time
dict construction.
"""
import sklearn.datasets as _skd

import numpy as _np


try:
    _skd.load_boston
except ImportError:
    def load_boston(*args, **kwargs):
        raise ImportError(
            "load_boston is deprecated and not supported in this project; "
            "the 'boston' bayesmark task is unused here."
        )
    _skd.load_boston = load_boston
    
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64

if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_