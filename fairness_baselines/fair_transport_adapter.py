"""Adapter for the ICML 2023 Wasserstein-barycenter Fair-transport baseline."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

import numpy as np


def check_fair_transport_dependencies(
    mip_solver: str = "AUTO", qp_solver: str = "AUTO"
) -> dict[str, Any]:
    """Validate the exact solver stack used by the archived ICML implementation."""
    try:
        import cvxpy
        import sklearn
        from third_party.fair_transport import PostProcessorDP
    except ImportError as exc:
        raise ImportError(
            "Fair-transport requires cvxpy, scikit-learn, and compatible LP/QP "
            "solvers. The active notebook kernel is "
            f"{sys.executable!r}. Select the 'Python (KTDLL Fairness)' kernel, "
            f"or run: \"{sys.executable}\" -m pip install cvxpy scikit-learn "
            "cylp osqp. See instruction_run.md."
        ) from exc
    del PostProcessorDP
    installed = tuple(sorted(cvxpy.installed_solvers()))
    lp_candidates = ("CBC", "CLARABEL", "SCIPY", "SCS")
    qp_candidates = ("OSQP", "CLARABEL", "SCS")
    resolved_mip = (
        next((name for name in lp_candidates if name in installed), None)
        if mip_solver == "AUTO"
        else mip_solver
    )
    resolved_qp = (
        next((name for name in qp_candidates if name in installed), None)
        if qp_solver == "AUTO"
        else qp_solver
    )
    missing = [
        requested
        for requested, resolved in (
            (f"LP:{mip_solver}", resolved_mip),
            (f"QP:{qp_solver}", resolved_qp),
        )
        if resolved not in installed
    ]
    if missing:
        raise RuntimeError(
            f"Fair-transport is missing CVXPY solver(s) {missing}; "
            f"installed solvers are {installed}. Install cylp and osqp for the "
            "upstream-preferred CBC/OSQP combination."
        )
    return {
        "cvxpy": cvxpy.__version__,
        "scikit_learn": sklearn.__version__,
        "installed_solvers": installed,
        "mip_solver": resolved_mip,
        "qp_solver": resolved_qp,
        "upstream_tag": "icml.23",
        "upstream_commit": "ff83c13c3c17de95ac7a29c0889727665014a08a",
    }


@dataclass(frozen=True)
class FairTransportConfig:
    alpha: float
    tolerance: float = 1e-8
    mip_solver: str = "AUTO"
    qp_solver: str = "AUTO"


@dataclass(frozen=True)
class FairTransportResult:
    prediction: np.ndarray
    diagnostics: dict[str, Any]


def _probability_matrix(value, name: str) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.ndim != 2 or probability.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(probability < -1e-8):
        raise ValueError(f"{name} contains materially negative probabilities.")
    probability = np.clip(probability, 0.0, None)
    mass = probability.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError(f"{name} contains a row with non-positive mass.")
    return probability / mass[:, None]


class FairTransportAdapter:
    """Fit the transport map on calibration scores and apply it to test scores."""

    def __init__(self, config: FairTransportConfig):
        if not np.isfinite(config.alpha) or config.alpha < 0:
            raise ValueError("Fair-transport alpha must be finite and non-negative.")
        if config.tolerance <= 0:
            raise ValueError("Fair-transport tolerance must be positive.")
        self.config = config
        self._processor = None
        self._group_to_code = None
        self._n_classes = None
        self._fit_diagnostics = None

    def _encode_groups(self, groups, name: str) -> np.ndarray:
        groups = np.asarray(groups)
        if groups.ndim != 1 or groups.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        unseen = set(np.unique(groups)) - set(self._group_to_code)
        if unseen:
            raise ValueError(f"{name} contains unseen sensitive groups: {unseen}.")
        return np.array([self._group_to_code[value] for value in groups], dtype=int)

    def fit(self, probability_calibration, sensitive_calibration):
        probability = _probability_matrix(
            probability_calibration, "probability_calibration"
        )
        sensitive = np.asarray(sensitive_calibration)
        if sensitive.ndim != 1 or len(sensitive) != len(probability):
            raise ValueError("Calibration scores and sensitive labels have inconsistent lengths.")
        unique_groups = np.sort(np.unique(sensitive))
        if len(unique_groups) < 2:
            raise ValueError("Fair-transport requires at least two sensitive groups.")
        self._group_to_code = {
            group: code for code, group in enumerate(unique_groups.tolist())
        }
        group_codes = self._encode_groups(sensitive, "sensitive_calibration")

        from third_party.fair_transport import PostProcessorDP

        started = time.perf_counter()
        processor = PostProcessorDP().fit(
            probability,
            group_codes,
            alpha=self.config.alpha,
            tol=self.config.tolerance,
            mip_solver=self.config.mip_solver,
            qp_solver=self.config.qp_solver,
        )
        runtime = time.perf_counter() - started
        q_by_group = np.asarray(processor.q_by_group_, dtype=float)
        achieved_calibration_gap = float(
            np.max(np.max(q_by_group, axis=0) - np.min(q_by_group, axis=0))
        )
        self._processor = processor
        self._n_classes = probability.shape[1]
        self._fit_diagnostics = {
            "status": "completed_without_exception",
            "convergence": str(processor.lp_status_),
            "map_status_by_group": processor.map_status_by_group_,
            "objective": float(processor.score_),
            "alpha": self.config.alpha,
            "calibration_dp_gap": achieved_calibration_gap,
            "mip_solver": processor.mip_solver_,
            "qp_solver": processor.qp_solver_,
            "postprocess_fit_runtime_seconds": runtime,
            "n_calibration": len(probability),
            "n_groups": len(unique_groups),
            "n_classes": self._n_classes,
        }
        return self

    def predict(self, probability, sensitive) -> np.ndarray:
        if self._processor is None:
            raise RuntimeError("Call fit() before predict().")
        probability = _probability_matrix(probability, "probability")
        if probability.shape[1] != self._n_classes:
            raise ValueError(
                f"Expected {self._n_classes} probability columns; "
                f"received {probability.shape[1]}."
            )
        group_codes = self._encode_groups(sensitive, "sensitive")
        if len(group_codes) != len(probability):
            raise ValueError("Prediction scores and sensitive labels have inconsistent lengths.")
        return self._processor.predict(probability, group_codes)

    def evaluate(self, probability, sensitive) -> FairTransportResult:
        started = time.perf_counter()
        prediction = self.predict(probability, sensitive)
        prediction_runtime = time.perf_counter() - started
        diagnostics = dict(self._fit_diagnostics)
        diagnostics["prediction_runtime_seconds"] = prediction_runtime
        diagnostics["total_runtime_seconds"] = (
            diagnostics["postprocess_fit_runtime_seconds"] + prediction_runtime
        )
        return FairTransportResult(prediction, diagnostics)
