"""
task_metadata.py — task-specific semantic context for LLM-based optimizers.

WHY THIS EXISTS
---------------
The three LLM optimizers (LLMOptimizer, LLAMBOOptimizer, LGBOOptimizer) used
to describe every task with the same abstract sentence ("Maximize an unknown
function over d variables. Bounds: ..."). The prompt-design PDFs (LLAMBO,
LGBO) show that grounding the prompt in *task semantics* — what the variables
mean, their real ranges/scales, the underlying objective, and (for real
experiments) the physical/chemical setting — is what makes the LLM useful.

This module is a single importable registry that maps a task's `.name`
(e.g. "branin", "bayesmark_SVM_wine", "yahpo_lcbench_3945", "lasso_diabetes",
"olympus_snar") to a `TaskContext` object. The optimizers call
`describe_task(task)` to obtain:

  * a one-line optimization objective / direction statement,
  * a coordinate-space note (what space the LLM proposes in),
  * a per-dimension parameter description block (positional, x0..x{d-1}),
  * an optional background paragraph (real experiments).

DESIGN CONSTRAINTS (important)
------------------------------
* The optimizers propose/parse raw coordinate vectors `x` of length `task.dim`,
  in the SAME normalized space the task's `evaluate()` expects. We therefore
  describe each dimension POSITIONALLY (x0, x1, ...) and NEVER instruct the LLM
  to emit named hyperparameters or real (unnormalized) values. The JSON output
  contract in every optimizer is unchanged.
* HPO/Lasso/YAHPO tasks operate in normalized [0,1]^d (Lasso in [-1,1]^d) and
  unscale internally. The parameter descriptions therefore explain the meaning
  and the underlying real range/scale, but the note tells the LLM to stay in
  the normalized coordinate space.
* Task files are NOT modified. Where a task already exposes structured space
  info (BayesmarkTask._space / YAHPOTask._space / *.param_description()), we
  reuse it via getattr; otherwise the hardcoded tables below are used.
* SYNTHETIC BLACK-BOX FUNCTIONS (branin, hartmann6, rastrigin, schwefel,
  gallagher) MUST remain genuinely black-box to the LLM. Their param_lines
  contain ONLY input dimension / range / scale. No commentary on optimum
  location, optimum value, modality, or landscape shape (e.g. "deceptive",
  "multimodal", "dominant peak", "three equivalent optima") is permitted,
  since that leaks solution-adjacent information the optimizer is meant to
  discover through evaluations alone and would bias any comparison against
  the non-LLM optimizers in the suite, which do not receive such hints.

The `TaskContext.render(...)` helper produces the final text; each optimizer
decides how to embed it (heading wording differs slightly per optimizer, but
the semantic content is identical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Core container
# --------------------------------------------------------------------------- #
@dataclass
class TaskContext:
    """Rendered semantic context for one task."""

    # High-level objective sentence, already phrased for the maximize
    # convention the whole suite uses ("higher evaluate() = better").
    objective: str

    # One line telling the LLM what coordinate space to propose in.
    coord_note: str

    # Per-dimension description lines, positional and in x-order. May be empty
    # (pure black-box synthetic where only bounds/scale matter).
    param_lines: List[str] = field(default_factory=list)

    # Optional real-world background paragraph (chemistry/materials/etc).
    background: Optional[str] = None

    def render(self, bounds, dim, include_background: bool = True) -> str:
        """
        Assemble the full task-description string.

        `bounds` / `dim` come from the live task so the numeric bounds are
        always exactly what the optimizer clips to, even if a task's dim is
        data-dependent (Lasso) or its ordering is emulator-defined (Olympus).
        """
        parts: List[str] = []
        if include_background and self.background:
            parts.append(self.background.strip())

        parts.append(self.objective.strip())

        bounds_str = ", ".join(
            f"x{i} in [{lo:.4g}, {hi:.4g}]" for i, (lo, hi) in enumerate(bounds)
        )
        parts.append(f"Search space: {dim} variables, bounds: {bounds_str}.")

        if self.param_lines:
            body = "\n".join(f"  - {ln}" for ln in self.param_lines)
            parts.append("Variable meanings (in coordinate order):\n" + body)

        parts.append(self.coord_note.strip())
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Precision note reused everywhere (kept from the original prompts)
# --------------------------------------------------------------------------- #
_PRECISION = (
    "Propose real-valued coordinates with several decimals of precision "
    "(e.g. 3.7182), not rounded integers."
)

_NORM_NOTE_01 = (
    "You propose coordinates DIRECTLY in the normalized [0,1] space (one value "
    "per variable); the benchmark maps them to the real ranges/scales described "
    "above. " + _PRECISION
)

_NORM_NOTE_PM1 = (
    "You propose coordinates DIRECTLY in the normalized [-1,1] space (one value "
    "per variable); the benchmark maps them internally. " + _PRECISION
)

_RAW_NOTE = _PRECISION


# --------------------------------------------------------------------------- #
# 1) Synthetic functions — black-box to the LLM: dimension/range/scale ONLY
# --------------------------------------------------------------------------- #
def _synthetic_context(name: str) -> Optional[TaskContext]:
    # IMPORTANT: keep these purely structural. Do not add landscape,
    # modality, or optimum-location/value commentary — see module docstring.
    if name == "branin":
        return TaskContext(
            objective=(
                "Objective: maximize an unknown 2D black-box function "
                "(higher is better)."
            ),
            coord_note=_RAW_NOTE,
            param_lines=[
                "x0: input dimension 0, range [-5, 10] (linear).",
                "x1: input dimension 1, range [0, 15] (linear).",
            ],
        )
    if name == "hartmann6":
        return TaskContext(
            objective=(
                "Objective: maximize an unknown 6D black-box function "
                "(higher is better)."
            ),
            coord_note=_RAW_NOTE,
            param_lines=[f"x{i}: input dimension {i}, range [0, 1] (linear)."
                         for i in range(6)],
        )
    if name == "rastrigin" or name.startswith("rastrigin_"):
        return TaskContext(
            objective=(
                "Objective: maximize an unknown black-box function "
                "(higher is better)."
            ),
            coord_note=_RAW_NOTE,
            param_lines=["all dimensions share range [-5.12, 5.12] (linear)."],
        )
    if name == "schwefel" or name.startswith("schwefel_"):
        return TaskContext(
            objective=(
                "Objective: maximize an unknown black-box function "
                "(higher is better)."
            ),
            coord_note=_RAW_NOTE,
            param_lines=["all dimensions share range [-500, 500] (linear)."],
        )
    if name == "gallagher" or name.startswith("gallagher_"):
        return TaskContext(
            objective=(
                "Objective: maximize an unknown black-box function "
                "(higher is better)."
            ),
            coord_note=_RAW_NOTE,
            param_lines=["all dimensions share range [-5, 5] (linear)."],
        )
    return None


# --------------------------------------------------------------------------- #
# 2) Bayesmark HPO — reuse the task's own _space where possible
# --------------------------------------------------------------------------- #
# Human-readable meaning per hyperparameter (keyed by Bayesmark param name).
_BAYESMARK_PARAM_MEANING: Dict[str, str] = {
    "C": "SVM inverse regularization strength (larger = less regularization)",
    "gamma": "RBF kernel coefficient (larger = more localized influence)",
    "tol": "optimization stopping tolerance",
    "max_depth": "maximum tree depth (larger = more complex trees)",
    "max_features": "fraction of features considered per split",
    "min_samples_split": "minimum samples required to split an internal node",
    "min_samples_leaf": "minimum samples required at a leaf node",
    "n_estimators": "number of trees / boosting stages in the ensemble",
    "bootstrap": "whether bootstrap samples are used (0=False, 1=True)",
    "min_impurity_decrease": "minimum impurity decrease required to split",
    "criterion": "split quality criterion (0=gini, 1=entropy)",
    "learning_rate": "shrinkage applied to each estimator's contribution",
    "hidden_layer_sizes": "number of units in the hidden layer",
    "alpha": "L2 regularization strength",
    "learning_rate_init": "initial SGD/Adam learning rate",
    "max_iter": "maximum training iterations",
    "beta_1": "Adam first-moment decay rate",
    "beta_2": "Adam second-moment decay rate",
}

_BAYESMARK_MODEL_CARD: Dict[str, str] = {
    "SVM": "a Support Vector Machine classifier",
    "RF": "a Random Forest classifier",
    "DT": "a Decision Tree classifier",
    "MLP-adam": "a multi-layer perceptron trained with Adam",
    "MLP-sgd": "a multi-layer perceptron trained with SGD",
    "ada": "an AdaBoost classifier",
    "kNN": "a k-nearest-neighbors classifier",
}

_BAYESMARK_DATA_CARD: Dict[str, str] = {
    "breast": "the Breast Cancer Wisconsin dataset (binary classification, "
              "~569 samples, 30 numerical features)",
    "digits": "the Digits dataset (10-class handwritten digit classification, "
              "~1797 samples, 64 pixel features)",
    "wine": "the Wine dataset (3-class classification, ~178 samples, "
            "13 numerical features)",
    "iris": "the Iris dataset (3-class classification, 150 samples, 4 features)",
    "diabetes": "the Diabetes regression dataset (~442 samples, 10 features)",
}


def _bayesmark_context(task, name: str) -> TaskContext:
    # name == "bayesmark_{model}_{dataset}"
    model = getattr(task, "model_name", None)
    dataset = getattr(task, "dataset", None)
    if model is None or dataset is None:
        # Fall back to parsing the canonical name.
        try:
            _, model, dataset = name.split("_", 2)
        except ValueError:
            model, dataset = "a model", "a dataset"

    model_card = _BAYESMARK_MODEL_CARD.get(model, f"a {model} model")
    data_card = _BAYESMARK_DATA_CARD.get(dataset, f"the {dataset} dataset")

    # Reuse the task's own structured space if present (name -> (lo, hi, scale)).
    space = getattr(task, "_space", None)
    param_names = getattr(task, "_param_names", None)
    lines: List[str] = []
    if space and param_names:
        for pname in param_names:
            lo, hi, scale = space[pname]
            meaning = _BAYESMARK_PARAM_MEANING.get(pname, pname)
            lines.append(
                f"{pname}: {meaning}; real range [{lo:g}, {hi:g}] ({scale} scale)."
            )

    objective = (
        f"Objective: tune the hyperparameters of {model_card} on {data_card} to "
        f"MAXIMIZE cross-validated performance (higher is better; accuracy for "
        f"classification, negative MSE for regression)."
    )
    return TaskContext(
        objective=objective,
        coord_note=_NORM_NOTE_01,
        param_lines=lines,
    )


# --------------------------------------------------------------------------- #
# 3) YAHPO lcbench HPO — reuse the task's own _space + rich meanings
# --------------------------------------------------------------------------- #
_LCBENCH_MEANING: Dict[str, str] = {
    "batch_size": "SGD mini-batch size",
    "learning_rate": "SGD learning rate",
    "momentum": "SGD momentum",
    "weight_decay": "L2 weight-decay regularization",
    "num_layers": "number of hidden layers",
    "max_units": "maximum units per hidden layer",
    "max_dropout": "maximum dropout rate",
}


def _yahpo_context(task, name: str) -> TaskContext:
    instance = getattr(task, "instance", None)
    objective_name = getattr(task, "objective", "val_accuracy")
    space = getattr(task, "_space", None)
    param_names = getattr(task, "_param_names", None)

    lines: List[str] = []
    if space and param_names:
        for pname in param_names:
            lo, hi, scale = space[pname]
            meaning = _LCBENCH_MEANING.get(pname, pname)
            lines.append(
                f"{pname}: {meaning}; real range [{lo:g}, {hi:g}] ({scale})."
            )

    inst_str = f"OpenML dataset {instance}" if instance else "an OpenML dataset"
    objective = (
        f"Objective: tune the training hyperparameters of a feed-forward neural "
        f"network on {inst_str} (YAHPO Gym lcbench surrogate) to MAXIMIZE "
        f"{objective_name} (higher is better)."
    )
    return TaskContext(
        objective=objective,
        coord_note=_NORM_NOTE_01,
        param_lines=lines,
    )


# --------------------------------------------------------------------------- #
# 4) LassoBench — real sparse-regression HPO
# --------------------------------------------------------------------------- #
_LASSO_DATA_CARD: Dict[str, str] = {
    "diabetes": "the diabetes dataset",
    "breast_cancer": "the breast-cancer dataset",
}


def _lasso_context(task, name: str) -> TaskContext:
    pick = getattr(task, "pick_data", None)
    if pick is None and name.startswith("lasso_"):
        pick = name[len("lasso_"):]
    data_card = _LASSO_DATA_CARD.get(pick, f"the {pick} dataset")

    objective = (
        f"Objective: choose the per-feature L1 (Lasso) regularization weights "
        f"for a sparse linear regression on {data_card} so as to MINIMIZE "
        f"cross-validated prediction loss. The benchmark reports negative loss, "
        f"so HIGHER is better here."
    )
    param_lines = [
        "each coordinate controls the log-scaled L1 penalty for one feature "
        "(more negative = weaker penalty / feature more free; more positive = "
        "stronger penalty / feature pushed toward zero).",
        "the search is high-dimensional; good solutions are typically sparse, "
        "penalizing most features while freeing a small informative subset.",
    ]
    return TaskContext(
        objective=objective,
        coord_note=_NORM_NOTE_PM1,
        param_lines=param_lines,
    )


# --------------------------------------------------------------------------- #
# 5) Olympus real-experiment emulators — hardcoded real chemistry semantics
# --------------------------------------------------------------------------- #
# Each entry: (background, real_objective_direction, [ (param_name, real_range,
# meaning) ... ] in the dataset's documented parameter order).
#
# NOTE ON DIRECTION: OlympusDatasetTask negates 'minimize' goals so that higher
# evaluate() is always better. We therefore state "MAXIMIZE (higher is better)"
# uniformly, but explain the underlying real objective (e.g. minimize E-factor)
# so the LLM understands the chemistry.
#
# NOTE ON ORDER: parameters are listed in the order documented in the Olympus
# paper (Hase et al. 2021, SI Tables V–XIV). The live task indexes x by the
# emulator's param_space; if a given emulator's order differs, only the
# per-variable label mapping would shift — coordinates and bounds still come
# from the live task, so evaluation is unaffected.

_OlympusEntry = Tuple[str, str, List[Tuple[str, str, str]]]

_OLYMPUS: Dict[str, _OlympusEntry] = {
    "olympus_snar": (
        "This is a real flow-chemistry experiment: a nucleophilic aromatic "
        "substitution (SNAr) reaction. The underlying goal is to minimize the "
        "E-factor (mass of waste per mass of product) by favoring the desired "
        "ortho product over para and bis adducts.",
        "Underlying real objective: minimize E-factor (less waste). Reported so "
        "that HIGHER is better.",
        [
            ("residence time", "[0.5, 2.0] min",
             "residence time in the flow apparatus"),
            ("morpholine equiv", "[1.0, 5.0]",
             "equivalents of the morpholine nucleophile"),
            ("concentration", "[0.1, 0.5] M", "reagent concentration"),
            ("temperature", "[60, 140] C", "reactor temperature"),
        ],
    ),
    "olympus_suzuki": (
        "This is a real palladium-catalyzed Suzuki-Miyaura carbon-carbon "
        "cross-coupling reaction run on an automated flow platform. The goal is "
        "to maximize product yield under mild conditions.",
        "Underlying real objective: maximize reaction yield (mol %). HIGHER is "
        "better.",
        [
            ("temperature", "[75, 90] C", "reaction temperature"),
            ("Pd mol", "[0.5, 5.0] mol %", "palladium catalyst loading"),
            ("ArBpin", "[1.0, 1.8] equiv",
             "equivalents of the pinacol boronate coupling partner"),
            ("K3PO4", "[1.5, 3.0] equiv",
             "equivalents of tripotassium phosphate base"),
        ],
    ),
    "olympus_benzylation": (
        "This is a real N-benzylation reaction in flow. The measured quantity is "
        "the yield of an undesired tertiary-amine impurity, which should be kept "
        "low while forming the desired secondary amine.",
        "Underlying real objective: minimize impurity yield. Reported so that "
        "HIGHER is better.",
        [
            ("flow rate", "[0.2, 0.4] mL/min", "reagent flow rate"),
            ("ratio", "[1, 5]",
             "benzyl bromide equivalents vs. the methylbenzylamine reagent"),
            ("solvent", "[0.5, 1.0]",
             "CHCl3 solvent equivalents vs. reagent stock solution"),
            ("temperature", "[110, 150] C", "reaction temperature"),
        ],
    ),
    "olympus_alkox": (
        "This is a real biocatalytic oxidation of benzyl alcohol by a copper "
        "radical oxidase (AlkOx). The goal is to maximize conversion by tuning "
        "enzyme/cocatalyst loadings and pH.",
        "Underlying real objective: maximize conversion. HIGHER is better.",
        [
            ("Catalase", "[0.05, 1.0] uM", "catalase concentration"),
            ("Horseradish peroxidase", "[0.5, 10.0] uM",
             "horseradish peroxidase concentration"),
            ("Alcohol oxidase", "[2.0, 8.0] nM",
             "alcohol oxidase concentration"),
            ("pH", "[6.0, 8.0]", "buffer pH"),
        ],
    ),
    "olympus_colors_bob": (
        "This is a real color-mixing experiment on the Bayesian Optimized "
        "Bartender (BOB) robot: five dyes are mixed and the resulting color is "
        "measured. The goal is to hit a target green RGB value.",
        "Underlying real objective: minimize distance to the target green color. "
        "Reported so that HIGHER is better.",
        [
            ("red", "[0, 1]", "amount of red dye"),
            ("orange", "[0, 1]", "amount of orange dye"),
            ("yellow", "[0, 1]", "amount of yellow dye"),
            ("blue", "[0, 1]", "amount of blue dye"),
            ("green", "[0, 1]", "amount of green dye"),
        ],
    ),
    "olympus_colors_n9": (
        "This is a real color-mixing experiment on an N9 robotic arm: three dyes "
        "are mixed and the color measured. The goal is to hit a target green RGB "
        "value.",
        "Underlying real objective: minimize distance to the target green color. "
        "Reported so that HIGHER is better.",
        [
            ("red", "[0, 1]", "amount of red dye"),
            ("green", "[0, 1]", "amount of green dye"),
            ("blue", "[0, 1]", "amount of blue dye"),
        ],
    ),
    "olympus_fullerenes": (
        "This is a real organic synthesis of o-xylenyl Buckminsterfullerene "
        "(C60) adducts. The goal is to maximize the mole fraction of the desired "
        "X1 + X2 adduct products.",
        "Underlying real objective: maximize mole fraction of desired adducts. "
        "HIGHER is better.",
        [
            ("reaction time", "[3, 31] min", "reaction time in the flow reactor"),
            ("sultine conc", "[1.5, 6.0]",
             "relative concentration of sultine to C60"),
            ("temperature", "[100, 150] C", "reaction temperature"),
        ],
    ),
    "olympus_hplc": (
        "This is a real calibration of an automated high-performance liquid "
        "chromatography (HPLC) system. The goal is to maximize the measured peak "
        "response.",
        "Underlying real objective: maximize peak area. HIGHER is better.",
        [
            ("sample loop", "[0.00, 0.08] mL", "sample-loop volume"),
            ("additional volume", "[0.00, 0.06] mL",
             "extra volume required to draw the sample"),
            ("tubing volume", "[0.1, 0.9] mL",
             "volume required to drive the sample"),
            ("sample flow", "[0.5, 2.5] mL/min", "sample-pump draw rate"),
            ("push speed", "[80, 150] Hz", "push-pump draw rate"),
            ("wait time", "[1, 10] s", "wait time"),
        ],
    ),
    "olympus_photo_pce10": (
        "This is a real materials experiment on the photostability of organic "
        "photovoltaic blends based on the PCE10 polymer. Each coordinate is the "
        "amount of one blend component; the goal is to minimize photodegradation.",
        "Underlying real objective: minimize photo-degradation. Reported so that "
        "HIGHER is better.",
        [
            ("mat 1", "[0, 1]", "amount of PCE10"),
            ("mat 2", "[0, 1]", "amount of P3HT"),
            ("mat 3", "[0, 1]", "amount of PCBM"),
            ("mat 4", "[0, 1]", "amount of oIDTBR"),
        ],
    ),
    "olympus_photo_wf3": (
        "This is a real materials experiment on the photostability of organic "
        "photovoltaic blends based on the WF3 polymer. Each coordinate is the "
        "amount of one blend component; the goal is to minimize photodegradation.",
        "Underlying real objective: minimize photo-degradation. Reported so that "
        "HIGHER is better.",
        [
            ("mat 1", "[0, 1]", "amount of WF3"),
            ("mat 2", "[0, 1]", "amount of P3HT"),
            ("mat 3", "[0, 1]", "amount of PCBM"),
            ("mat 4", "[0, 1]", "amount of oIDTBR"),
        ],
    ),
}


def _olympus_context(task, name: str) -> TaskContext:
    entry = _OLYMPUS.get(name)
    if entry is None:
        # Unknown olympus dataset: degrade gracefully to abstract description.
        return TaskContext(
            objective=(
                "Objective: maximize the response of a real chemistry/materials "
                "experiment emulated by a Bayesian neural network (higher is "
                "better)."
            ),
            coord_note=_RAW_NOTE,
        )
    background, direction, params = entry
    param_lines = [
        f"{pname}: {meaning}; real range {rng}." for (pname, rng, meaning) in params
    ]
    objective = (
        "Objective: this is a REAL experimental response surface (emulated by a "
        "Bayesian neural network trained on measured data). MAXIMIZE the reported "
        "value (higher is better). " + direction
    )
    return TaskContext(
        objective=objective,
        coord_note=(
            "You propose the real experimental settings DIRECTLY, one value per "
            "variable, within the ranges above. " + _PRECISION
        ),
        param_lines=param_lines,
        background=background,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_task_context(task) -> TaskContext:
    """
    Resolve a TaskContext for a live task object via its `.name`.

    Robust to unknown tasks: returns a generic black-box context so callers
    never crash. Never raises.
    """
    name = getattr(task, "name", "") or ""

    # Synthetic (exact or dim-suffixed like "rastrigin_20").
    syn = _synthetic_context(name)
    if syn is not None:
        return syn

    if name.startswith("bayesmark_"):
        return _bayesmark_context(task, name)
    if name.startswith("yahpo_"):
        return _yahpo_context(task, name)
    if name.startswith("lasso_"):
        return _lasso_context(task, name)
    if name.startswith("olympus_"):
        return _olympus_context(task, name)

    # Generic fallback: abstract black-box, real bounds filled in by render().
    return TaskContext(
        objective="Objective: maximize an unknown black-box function (higher is "
                  "better).",
        coord_note=_RAW_NOTE,
    )


def describe_task(task, include_background: bool = True) -> str:
    """
    Convenience: return the fully rendered task-description string for `task`,
    using the task's live bounds/dim. This is the single call the optimizers
    make in place of their old `_task_description()` body.
    """
    ctx = get_task_context(task)
    return ctx.render(task.bounds, task.dim, include_background=include_background)
