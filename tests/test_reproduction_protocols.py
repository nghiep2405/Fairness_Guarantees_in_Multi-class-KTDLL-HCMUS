import tempfile
import unittest

import numpy as np

from reproduction_protocols import (
    BinaryProtocolConfig,
    SyntheticProtocolConfig,
    run_binary_real_protocol,
    run_synthetic_figure1,
    run_synthetic_figure4,
    run_synthetic_figures2_3,
)


def argmax_postprocess(
    probability_test,
    probability_pool,
    sensitive_test,
    sensitive_pool,
    epsilon,
    random_state=None,
):
    del probability_pool, sensitive_test, sensitive_pool, epsilon, random_state
    return np.argmax(probability_test, axis=1)


def binary_loader(K, seed):
    if K != 2:
        raise ValueError
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(300, 4))
    sensitive = rng.choice([-1, 1], size=300)
    labels = (features[:, 0] + 0.3 * sensitive > 0).astype(int)
    data = np.column_stack([features, sensitive])
    return (
        data[:180],
        data[180:240],
        labels[:180],
        labels[180:240],
        data[240:],
    )


class ReproductionProtocolTests(unittest.TestCase):
    def test_synthetic_repetitions_are_aggregated(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = SyntheticProtocolConfig(
                n_repetitions=2,
                n_samples=600,
                n_classes=3,
                n_features=4,
                n_clusters=2,
                p_values=(0.5, 0.75),
                epsilons=(0.0, 0.1),
                output_dir=output_dir,
            )
            figure1_raw, figure1_summary = run_synthetic_figure1(config)
            figure2_raw, figure2_summary, figure3_raw, figure3_summary = (
                run_synthetic_figures2_3(argmax_postprocess, config)
            )
            figure4_raw, figure4_summary = run_synthetic_figure4(
                argmax_postprocess, config
            )
            self.assertEqual(len(figure1_raw), 4)
            self.assertTrue((figure1_summary["Repetitions"] == 2).all())
            self.assertTrue((figure2_summary["Repetitions"] == 2).all())
            self.assertTrue((figure3_summary["Repetitions"] == 2).all())
            self.assertTrue((figure4_summary["Repetitions"] == 2).all())
            self.assertFalse(figure2_raw.empty)
            self.assertFalse(figure3_raw.empty)
            self.assertFalse(figure4_raw.empty)

    def test_binary_fairlearn_grid_maps_to_tolerance_field(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = BinaryProtocolConfig(
                n_repetitions=1,
                epsilons=(0.0,),
                fairlearn_tolerances=(0.0001,),
                tuning_n_iter=1,
                cv_folds=2,
                run_adversarial=False,
                output_dir=output_dir,
            )
            raw, summary = run_binary_real_protocol(
                "TEST", binary_loader, argmax_postprocess, config
            )
            fairlearn = raw[raw["Method"] == "Fairlearn"]
            self.assertEqual(len(fairlearn), 3)
            self.assertTrue((fairlearn["FairlearnTolerance"] == 0.0001).all())
            self.assertTrue(fairlearn["Epsilon"].isna().all())
            self.assertFalse(summary.empty)


if __name__ == "__main__":
    unittest.main()
