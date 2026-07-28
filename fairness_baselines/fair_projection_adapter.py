"""Safe adapter around the upstream FairProjection implementation.

The adapter deliberately reuses an already-fitted base estimator.  The upstream
``GFair.fit`` method normally fits the estimator again; wrapping it in
``_FrozenProbabilityEstimator`` turns that call into a no-op while preserving
the exact ``predict_proba`` output shared by all methods in the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

import numpy as np


def ensure_fair_projection_runtime(method: str = "tf") -> dict[str, Any]:
    """Validate process-wide requirements before starting a projection."""
    if method not in {"tf", "np"}:
        raise ValueError("FairProjection method must be 'tf' or 'np'.")
    if method == "np":
        return {"method": method, "tensorflow_eager_enabled": None}
    import tensorflow

    eager_enabled = bool(tensorflow.executing_eagerly())
    if not eager_enabled:
        raise RuntimeError(
            "FairProjection method='tf' requires TensorFlow eager execution, but "
            "eager mode is disabled in this process. Restart the notebook kernel "
            "and run AIF360 through the isolated subprocess; or select method='np'."
        )
    return {"method": method, "tensorflow_eager_enabled": eager_enabled}


def check_fair_projection_dependencies(method: str = "tf") -> dict[str, Any]:
    """Import the solver stack once and fail before an expensive experiment."""
    try:
        import cvxpy
        import scipy
        import tqdm
        from third_party.fair_projection.GroupFair import GFair
        if method == "tf":
            import tensorflow
    except ImportError as exc:
        tensorflow_requirement = "tensorflow, " if method == "tf" else ""
        raise ImportError(
            f"FairProjection method={method!r} requires "
            f"{tensorflow_requirement}cvxpy, scipy and tqdm. "
            f"The active notebook kernel is {sys.executable!r}. Select the "
            "'Python (KTDLL Fairness)' kernel, or install into the active kernel "
            f"with: \"{sys.executable}\" -m pip install tensorflow cvxpy scipy tqdm. "
            "See README.md."
        ) from exc
    del GFair
    versions = {
        "cvxpy": cvxpy.__version__,
        "scipy": scipy.__version__,
        "tqdm": tqdm.__version__,
    }
    if method == "tf":
        versions["tensorflow"] = tensorflow.__version__
    versions.update(ensure_fair_projection_runtime(method))
    return versions


@dataclass(frozen=True)
class FairProjectionConfig:
    alpha: float
    divergence: str = "cross-entropy"
    constraint: str = "sp"
    rho: float = 2.0
    max_iter: int = 500
    method: str = "tf"
    verbose: bool = False


@dataclass(frozen=True)
class FairProjectionResult:
    probability: np.ndarray
    prediction: np.ndarray
    diagnostics: dict[str, Any]


class _FrozenProbabilityEstimator:
    """Expose a fitted estimator to GFair without allowing it to be refit."""

    def __init__(self, estimator: Any):
        if not hasattr(estimator, "predict_proba"):
            raise TypeError("FairProjection requires a fitted estimator with predict_proba().")
        self.estimator = estimator
        self.classes_ = np.asarray(estimator.classes_)

    def fit(self, X, y, sample_weight=None):
        del X, y, sample_weight
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def _as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def _as_feature_matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite two-dimensional matrix.")
    return array


def _normalize_probability(value, expected_rows: int, expected_classes: int) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.ndim == 3 and probability.shape[-1] == 1:
        probability = probability[..., 0]
    if probability.shape != (expected_rows, expected_classes):
        raise ValueError(
            "FairProjection returned probability shape "
            f"{probability.shape}; expected {(expected_rows, expected_classes)}."
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("FairProjection returned non-finite probabilities.")
    if np.min(probability) < -1e-8:
        raise ValueError("FairProjection returned materially negative probabilities.")
    probability = np.clip(probability, 0.0, None)
    row_sum = probability.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError("FairProjection returned a row with non-positive probability mass.")
    probability = probability / row_sum[:, None]
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("Could not normalize FairProjection probabilities.")
    return probability


class FairProjectionAdapter:
    """Fit FairProjection on calibration data and project held-out probabilities."""

    def __init__(self, base_estimator: Any, config: FairProjectionConfig):
        if not np.isfinite(config.alpha) or config.alpha < 0:
            raise ValueError("FairProjection alpha must be finite and non-negative.")
        if config.constraint != "sp":
            raise ValueError("This reproduction adapter supports Statistical Parity ('sp') only.")
        if config.divergence not in {"cross-entropy", "kl"}:
            raise ValueError("Unsupported FairProjection divergence.")
        if config.method not in {"tf", "np"}:
            raise ValueError("FairProjection method must be 'tf' or 'np'.")
        if config.max_iter <= 0 or config.rho <= 0:
            raise ValueError("FairProjection max_iter and rho must be positive.")
        self.base_estimator = base_estimator
        self.config = config
        self._gf = None
        self._n_classes = None
        self._fit_diagnostics = None

    def fit(self, X_train, y_train, sensitive_train, X_calibration, sensitive_calibration):
        ensure_fair_projection_runtime(self.config.method)
        X_train = _as_feature_matrix(X_train, "X_train")
        X_calibration = _as_feature_matrix(X_calibration, "X_calibration")
        y_train = _as_vector(y_train, "y_train").astype(int)
        sensitive_train = _as_vector(sensitive_train, "sensitive_train")
        sensitive_calibration = _as_vector(
            sensitive_calibration, "sensitive_calibration"
        )
        if len(y_train) != len(X_train) or len(sensitive_train) != len(X_train):
            raise ValueError("Training arrays have inconsistent lengths.")
        if len(sensitive_calibration) != len(X_calibration):
            raise ValueError("Calibration arrays have inconsistent lengths.")
        if len(np.unique(sensitive_train)) < 2 or len(np.unique(sensitive_calibration)) < 2:
            raise ValueError("Training and calibration must each contain at least two groups.")

        class_order = np.asarray(self.base_estimator.classes_)
        expected_order = np.arange(len(class_order))
        if not np.array_equal(class_order, expected_order):
            raise ValueError(
                f"FairProjection requires class order 0..K-1; got {class_order}."
            )
        if not np.array_equal(np.unique(y_train), expected_order):
            raise ValueError("Training partition does not contain every expected class.")
        self._n_classes = len(class_order)

        # Lazy import prevents TensorFlow/CVXPY from loading when the baseline is disabled.
        from third_party.fair_projection.GroupFair import GFair

        frozen = _FrozenProbabilityEstimator(self.base_estimator)
        gf = GFair(
            frozen,
            clf_S=None,
            clf_SgY=None,
            div=self.config.divergence,
        )
        started = time.perf_counter()
        gf.fit(X_train, y_train, sensitive_train, sample_weight=None)
        gf.project(
            X_calibration,
            s=sensitive_calibration,
            constraints=[(self.config.constraint, self.config.alpha)],
            rho=self.config.rho,
            max_iter=self.config.max_iter,
            method=self.config.method,
            verbose=self.config.verbose,
        )
        runtime = time.perf_counter() - started
        if not getattr(gf, "Projected", False):
            raise RuntimeError("FairProjection core returned without marking the model projected.")
        dual = np.asarray(gf.l)
        if not np.all(np.isfinite(dual)):
            raise RuntimeError("FairProjection produced non-finite dual parameters.")

        self._gf = gf
        self._fit_diagnostics = {
            "status": "completed_without_exception",
            "convergence": "convergence_not_observable",
            "solver_residual": np.nan,
            "alpha": self.config.alpha,
            "constraint": self.config.constraint,
            "divergence": self.config.divergence,
            "rho": self.config.rho,
            "max_iter_requested": self.config.max_iter,
            "method": self.config.method,
            "projection_fit_runtime_seconds": runtime,
            "dual_shape": tuple(dual.shape),
            "dual_max": float(np.max(dual)),
            "dual_min": float(np.min(dual)),
        }
        return self

    def predict_proba(self, X, sensitive) -> np.ndarray:
        if self._gf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = _as_feature_matrix(X, "X")
        sensitive = _as_vector(sensitive, "sensitive")
        if len(X) != len(sensitive):
            raise ValueError("X and sensitive have inconsistent lengths.")
        raw = self._gf.predict_proba(X, s=sensitive)
        return _normalize_probability(raw, len(X), self._n_classes)

    def predict(self, X, sensitive) -> np.ndarray:
        return np.argmax(self.predict_proba(X, sensitive), axis=1)

    def evaluate(self, X, sensitive) -> FairProjectionResult:
        started = time.perf_counter()
        probability = self.predict_proba(X, sensitive)
        prediction = np.argmax(probability, axis=1)
        predict_runtime = time.perf_counter() - started
        diagnostics = dict(self._fit_diagnostics)
        diagnostics["prediction_runtime_seconds"] = predict_runtime
        diagnostics["total_runtime_seconds"] = (
            diagnostics["projection_fit_runtime_seconds"] + predict_runtime
        )
        return FairProjectionResult(probability, prediction, diagnostics)
