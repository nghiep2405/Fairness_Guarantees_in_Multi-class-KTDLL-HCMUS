"""Isolated TensorFlow v1 worker for the AIF360 adversarial baseline.

This module is intentionally executed with ``python -m`` in a child process.
AIF360's adversarial debiasing requires TensorFlow v1 graph mode, which is an
irreversible process-wide setting and must not leak into the notebook kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd


def _as_aif360_dataset(X, y, sensitive):
    from aif360.datasets import BinaryLabelDataset

    frame = pd.DataFrame(X)
    frame["label"] = y
    frame["sensitive_attr"] = np.where(sensitive == -1, 0, 1)
    return BinaryLabelDataset(
        df=frame,
        label_names=["label"],
        protected_attribute_names=["sensitive_attr"],
        favorable_label=1,
        unfavorable_label=0,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope-name", default="adversarial_debiasing")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=50)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    if args.probe:
        np.savez_compressed(
            args.output,
            eager_enabled=np.asarray(tf.executing_eagerly()),
        )
        return 0
    if args.input is None:
        raise ValueError("--input is required unless --probe is used.")

    from aif360.algorithms.inprocessing import AdversarialDebiasing

    with np.load(args.input, allow_pickle=False) as payload:
        X_train = payload["X_train"]
        y_train = payload["y_train"]
        sensitive_train = payload["sensitive_train"]
        X_test = payload["X_test"]
        y_test = payload["y_test"]
        sensitive_test = payload["sensitive_test"]

    np.random.seed(args.seed)
    tf.reset_default_graph()
    tf.set_random_seed(args.seed)
    training = _as_aif360_dataset(X_train, y_train, sensitive_train)
    testing = _as_aif360_dataset(X_test, y_test, sensitive_test)
    session = tf.Session()
    try:
        model = AdversarialDebiasing(
            privileged_groups=[{"sensitive_attr": 1}],
            unprivileged_groups=[{"sensitive_attr": 0}],
            scope_name=args.scope_name,
            debias=True,
            sess=session,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            classifier_num_hidden_units=args.hidden_units,
            adversary_loss_weight=args.weight,
        )
        started = time.perf_counter()
        model.fit(training)
        prediction = model.predict(testing).labels.ravel()
        runtime = time.perf_counter() - started
    finally:
        session.close()

    np.savez_compressed(
        args.output,
        prediction=np.asarray(prediction),
        runtime_seconds=np.asarray(runtime),
        eager_enabled=np.asarray(tf.executing_eagerly()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
