"""Parent-process interface for the isolated AIF360 graph-mode worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AIF360AdversarialConfig:
    adversary_loss_weight: float
    seed: int
    scope_name: str
    num_epochs: int = 200
    batch_size: int = 128
    hidden_units: int = 50


@dataclass(frozen=True)
class AIF360AdversarialResult:
    prediction: np.ndarray
    runtime_seconds: float
    diagnostics: dict[str, Any]


def _worker_environment(seed: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    environment["PYTHONHASHSEED"] = str(seed)
    return environment


def _run_worker(arguments, seed: int, timeout_seconds: float):
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "fairness_baselines.aif360_worker", *arguments],
        cwd=repository_root,
        env=_worker_environment(seed),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "The isolated AIF360 worker failed "
            f"(exit code {completed.returncode}).\n{detail}"
        )
    return completed


def _as_finite_array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def run_aif360_adversarial_subprocess(
    X_train,
    y_train,
    sensitive_train,
    X_test,
    y_test,
    sensitive_test,
    config: AIF360AdversarialConfig,
    timeout_seconds: float = 3600,
) -> AIF360AdversarialResult:
    """Fit/predict with AIF360 without changing TensorFlow mode in the parent."""
    X_train = _as_finite_array(X_train, "X_train", 2)
    y_train = _as_finite_array(y_train, "y_train", 1)
    sensitive_train = _as_finite_array(sensitive_train, "sensitive_train", 1)
    X_test = _as_finite_array(X_test, "X_test", 2)
    y_test = _as_finite_array(y_test, "y_test", 1)
    sensitive_test = _as_finite_array(sensitive_test, "sensitive_test", 1)
    if len(X_train) != len(y_train) or len(X_train) != len(sensitive_train):
        raise ValueError("Training arrays have inconsistent lengths.")
    if len(X_test) != len(y_test) or len(X_test) != len(sensitive_test):
        raise ValueError("Test arrays have inconsistent lengths.")
    if config.num_epochs <= 0 or config.batch_size <= 0 or config.hidden_units <= 0:
        raise ValueError("AIF360 epochs, batch size and hidden units must be positive.")
    if not np.isfinite(config.adversary_loss_weight):
        raise ValueError("AIF360 adversary loss weight must be finite.")

    with tempfile.TemporaryDirectory(prefix="aif360-worker-") as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "output.npz"
        np.savez_compressed(
            input_path,
            X_train=X_train,
            y_train=y_train,
            sensitive_train=sensitive_train,
            X_test=X_test,
            y_test=y_test,
            sensitive_test=sensitive_test,
        )
        completed = _run_worker(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--weight",
                str(config.adversary_loss_weight),
                "--seed",
                str(config.seed),
                "--scope-name",
                config.scope_name,
                "--num-epochs",
                str(config.num_epochs),
                "--batch-size",
                str(config.batch_size),
                "--hidden-units",
                str(config.hidden_units),
            ],
            config.seed,
            timeout_seconds,
        )
        if not output_path.exists():
            raise RuntimeError("The isolated AIF360 worker produced no output artifact.")
        with np.load(output_path, allow_pickle=False) as output:
            prediction = np.asarray(output["prediction"]).ravel()
            runtime = float(output["runtime_seconds"])
            worker_eager = bool(output["eager_enabled"])

    if len(prediction) != len(X_test) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("The isolated AIF360 worker returned invalid predictions.")
    diagnostics = {
        "worker_eager_enabled": worker_eager,
        "worker_stdout": completed.stdout.strip(),
        "worker_stderr": completed.stderr.strip(),
    }
    return AIF360AdversarialResult(prediction, runtime, diagnostics)


def probe_aif360_subprocess_runtime(timeout_seconds: float = 120) -> dict[str, bool]:
    """Return the child's TensorFlow mode for a lightweight isolation check."""
    with tempfile.TemporaryDirectory(prefix="aif360-probe-") as temp_dir:
        output_path = Path(temp_dir) / "probe.npz"
        _run_worker(
            ["--output", str(output_path), "--probe"],
            seed=0,
            timeout_seconds=timeout_seconds,
        )
        with np.load(output_path, allow_pickle=False) as output:
            return {"worker_eager_enabled": bool(output["eager_enabled"])}
