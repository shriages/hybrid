"""
Compatibility shim for ConfigSpace version drift.
Import this BEFORE importing yahpo_gym anywhere.
(Then in whatever module imports yahpo, import the shim first)
"""
import ConfigSpace
from ConfigSpace import ConfigurationSpace

# Example: if a method/attribute was removed and you can safely stub it.
# Only do this if the shimmed behavior is actually correct/unused —
# faking _sort_hyperparameters is NOT safe, so this is illustrative.
if not hasattr(ConfigurationSpace, "_sort_hyperparameters"):
    def _sort_hyperparameters(self):
        # No-op only works if callers don't depend on the side effect.
        return None
    ConfigurationSpace._sort_hyperparameters = _sort_hyperparameters