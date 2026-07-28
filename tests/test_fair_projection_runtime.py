import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from fairness_baselines.aif360_subprocess import (
    AIF360AdversarialConfig,
    probe_aif360_subprocess_runtime,
    run_aif360_adversarial_subprocess,
)
from fairness_baselines.fair_projection_adapter import (
    FairProjectionAdapter,
    FairProjectionConfig,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
HAS_AIF360 = importlib.util.find_spec("aif360") is not None


def _subprocess_environment():
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    return environment


class FairProjectionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_samples=150,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=9,
        )
        cls.X = X
        cls.y = y
        cls.sensitive = np.tile(np.array([-1, 1]), 75)
        cls.model = LogisticRegression(max_iter=1000, random_state=9).fit(
            X[:90], y[:90]
        )

    def _fit_adapter(self, method):
        adapter = FairProjectionAdapter(
            self.model,
            FairProjectionConfig(
                alpha=0.1,
                max_iter=1,
                method=method,
            ),
        )
        adapter.fit(
            self.X[:90],
            self.y[:90],
            self.sensitive[:90],
            self.X[90:120],
            self.sensitive[90:120],
        )
        return adapter

    def test_numpy_backend_returns_normalized_multiclass_probabilities(self):
        probability = self._fit_adapter("np").predict_proba(
            self.X[120:], self.sensitive[120:]
        )
        self.assertEqual(probability.shape, (30, 3))
        self.assertTrue(np.all(np.isfinite(probability)))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-7)

    def test_numpy_backend_does_not_import_tensorflow(self):
        code = (
            "import sys; "
            "from third_party.fair_projection.GroupFair import GFair; "
            "assert GFair is not None; "
            "assert 'tensorflow' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            env=_subprocess_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
    def test_tensorflow_backend_runs_in_eager_mode(self):
        import tensorflow as tf

        self.assertTrue(tf.executing_eagerly())
        probability = self._fit_adapter("tf").predict_proba(
            self.X[120:], self.sensitive[120:]
        )
        self.assertEqual(probability.shape, (30, 3))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-7)
        self.assertTrue(tf.executing_eagerly())

    @unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
    def test_tensorflow_backend_fails_fast_in_graph_mode(self):
        code = "\n".join(
            [
                "import tensorflow.compat.v1 as tf",
                "tf.disable_eager_execution()",
                (
                    "from fairness_baselines.fair_projection_adapter "
                    "import ensure_fair_projection_runtime"
                ),
                "ensure_fair_projection_runtime('tf')",
            ]
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            env=_subprocess_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires TensorFlow eager execution", completed.stderr)


class AIF360IsolationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is not installed")
    def test_graph_mode_is_confined_to_worker(self):
        import tensorflow as tf

        self.assertTrue(tf.executing_eagerly())
        diagnostics = probe_aif360_subprocess_runtime()
        self.assertFalse(diagnostics["worker_eager_enabled"])
        self.assertTrue(tf.executing_eagerly())

    @unittest.skipUnless(
        HAS_TENSORFLOW and HAS_AIF360,
        "TensorFlow and AIF360 are required",
    )
    def test_adversarial_worker_returns_prediction_and_runtime(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(48, 3))
        sensitive = np.tile(np.array([-1, 1]), 24)
        y = (X[:, 0] + 0.4 * sensitive > 0).astype(int)
        result = run_aif360_adversarial_subprocess(
            X[:40],
            y[:40],
            sensitive[:40],
            X[40:],
            y[40:],
            sensitive[40:],
            AIF360AdversarialConfig(
                adversary_loss_weight=0.1,
                seed=7,
                scope_name="test_worker",
                num_epochs=1,
                batch_size=8,
                hidden_units=4,
            ),
            timeout_seconds=120,
        )
        self.assertEqual(result.prediction.shape, (8,))
        self.assertGreaterEqual(result.runtime_seconds, 0.0)
        self.assertFalse(result.diagnostics["worker_eager_enabled"])


if __name__ == "__main__":
    unittest.main()
