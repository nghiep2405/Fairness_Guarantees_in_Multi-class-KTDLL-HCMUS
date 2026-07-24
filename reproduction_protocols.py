"""Paper-style repeated experiment protocols.

The notebook delegates long-running synthetic and binary experiments to this
module so raw repetitions, summary statistics, seeds, and artifacts are
handled consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


PostprocessFn = Callable[..., np.ndarray]


def empirical_unfairness(y_pred, sensitive, classes) -> float:
    """Maximum pairwise empirical demographic-parity gap."""
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    groups = np.unique(sensitive)
    if len(groups) < 2:
        raise ValueError("At least two sensitive groups are required.")
    return float(
        max(
            max(np.mean(y_pred[sensitive == group] == label) for group in groups)
            - min(np.mean(y_pred[sensitive == group] == label) for group in groups)
            for label in classes
        )
    )


def summarize_repetitions(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate report metrics while retaining the repetition count."""
    return (
        raw.groupby(keys, dropna=False)
        .agg(
            AccuracyMean=("Accuracy", "mean"),
            AccuracyStd=("Accuracy", "std"),
            UnfairnessMean=("Unfairness", "mean"),
            UnfairnessStd=("Unfairness", "std"),
            RuntimeMean=("RuntimeSeconds", "mean"),
            RuntimeStd=("RuntimeSeconds", "std"),
            Repetitions=("Repetition", "nunique"),
        )
        .reset_index()
    )


def _save_protocol_artifacts(
    output_dir: str,
    prefix: str,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    config,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw.to_csv(directory / f"{prefix}_raw.csv", index=False)
    summary.to_csv(directory / f"{prefix}_summary.csv", index=False)
    pd.DataFrame([asdict(config)]).to_json(
        directory / f"{prefix}_config.json", orient="records", indent=2
    )


@dataclass(frozen=True)
class SyntheticProtocolConfig:
    n_repetitions: int = 30
    master_seed: int = 777
    n_samples: int = 5000
    n_classes: int = 6
    n_features: int = 20
    n_clusters: int = 10
    p_values: tuple = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99)
    epsilons: tuple = (0.0, 0.05, 0.1, 0.15, 0.2)
    figure10_epsilons: tuple = (0.0, 0.05, 0.1, 0.15)
    figure10_train_sizes: tuple = tuple(range(100, 2001, 100))
    figure10_pool_sizes: tuple = tuple(range(20, 301, 20))
    figure10_fixed_pool_size: int = 4000
    figure10_fixed_train_size: int = 2000
    figure10_test_size: int = 4000
    figure10_oracle_pool_size: int = 20000
    output_dir: str = "outputs/synthetic"


@dataclass(frozen=True)
class SyntheticDistribution:
    """Parameters of the paper's Gaussian-mixture synthetic distribution."""

    means: np.ndarray
    p: float

    @property
    def n_classes(self) -> int:
        return self.means.shape[0]

    @property
    def n_clusters(self) -> int:
        return self.means.shape[1]

    @property
    def n_features(self) -> int:
        return self.means.shape[2]

    def sample(self, n_samples: int, rng: np.random.Generator):
        y = rng.integers(0, self.n_classes, size=n_samples)
        component = rng.integers(0, self.n_clusters, size=n_samples)
        X = self.means[y, component] + rng.normal(
            size=(n_samples, self.n_features)
        )
        prob_positive = np.where(
            y < self.n_classes // 2, self.p, 1.0 - self.p
        )
        sensitive = 2 * rng.binomial(1, prob_positive) - 1
        return np.column_stack([X, sensitive]), y.astype(int)

    def posterior(self, X_with_sensitive: np.ndarray) -> np.ndarray:
        """Compute the exact P(Y=k|X,S) under the known synthetic model."""
        X_with_sensitive = np.asarray(X_with_sensitive, dtype=float)
        X = X_with_sensitive[:, :-1]
        sensitive = X_with_sensitive[:, -1].astype(int)
        n = len(X)
        log_score = np.empty((n, self.n_classes), dtype=float)
        normal_constant = -0.5 * self.n_features * np.log(2 * np.pi)
        for label in range(self.n_classes):
            difference = X[:, None, :] - self.means[label][None, :, :]
            component_log_density = (
                normal_constant - 0.5 * np.sum(difference**2, axis=2)
            )
            log_mixture = (
                logsumexp(component_log_density, axis=1)
                - np.log(self.n_clusters)
            )
            positive_probability = (
                self.p if label < self.n_classes // 2 else 1.0 - self.p
            )
            sensitive_probability = np.where(
                sensitive == 1,
                positive_probability,
                1.0 - positive_probability,
            )
            log_score[:, label] = log_mixture + np.log(
                np.clip(sensitive_probability, 1e-15, 1.0)
            )
        log_score -= logsumexp(log_score, axis=1, keepdims=True)
        return np.exp(log_score)


def make_synthetic_distribution(
    rng: np.random.Generator,
    n_classes: int,
    n_features: int,
    n_clusters: int,
    p: float,
) -> SyntheticDistribution:
    centroids = rng.uniform(-1, 1, size=(n_classes, n_features))
    offsets = rng.normal(size=(n_classes, n_clusters, n_features))
    return SyntheticDistribution(centroids[:, None, :] + offsets, p)


def _split_sample(distribution, rng, train_size, test_size, pool_size):
    data, labels = distribution.sample(train_size + test_size + pool_size, rng)
    train_stop = train_size
    test_stop = train_size + test_size
    return (
        data[:train_stop],
        labels[:train_stop],
        data[train_stop:test_stop],
        labels[train_stop:test_stop],
        data[test_stop:],
        labels[test_stop:],
    )


def run_synthetic_figure1(config=SyntheticProtocolConfig()):
    """Bayes-classifier unfairness versus p, with 30-run mean/std."""
    records = []
    for repetition in range(config.n_repetitions):
        distribution_seed = config.master_seed + 10000 * repetition
        distribution_rng = np.random.default_rng(distribution_seed)
        base_distribution = make_synthetic_distribution(
            distribution_rng,
            config.n_classes,
            config.n_features,
            config.n_clusters,
            0.5,
        )
        for p_index, p in enumerate(config.p_values):
            seed = distribution_seed + p_index + 1
            rng = np.random.default_rng(seed)
            distribution = SyntheticDistribution(base_distribution.means, p)
            test, y_test = distribution.sample(config.n_samples, rng)
            probability = distribution.posterior(test)
            prediction = np.argmax(probability, axis=1)
            records.append(
                {
                    "Figure": 1,
                    "Repetition": repetition,
                    "Seed": seed,
                    "DistributionSeed": distribution_seed,
                    "p": p,
                    "Accuracy": accuracy_score(y_test, prediction),
                    "Unfairness": empirical_unfairness(
                        prediction, test[:, -1], np.arange(config.n_classes)
                    ),
                    "RuntimeSeconds": 0.0,
                }
            )
    raw = pd.DataFrame(records)
    summary = summarize_repetitions(raw, ["Figure", "p"])
    _save_protocol_artifacts(config.output_dir, "figure1", raw, summary, config)
    return raw, summary


def run_synthetic_figures2_3(
    postprocess_fn: PostprocessFn, config=SyntheticProtocolConfig()
):
    """Paper Figures 2-3 using 60/20/20 splits over 30 repetitions."""
    records = []
    distribution_records = []
    train_size = int(0.60 * config.n_samples)
    test_size = int(0.20 * config.n_samples)
    pool_size = config.n_samples - train_size - test_size

    for repetition in range(config.n_repetitions):
        distribution_seed = config.master_seed + 10000 * repetition
        distribution_rng = np.random.default_rng(distribution_seed)
        base_distribution = make_synthetic_distribution(
            distribution_rng,
            config.n_classes,
            config.n_features,
            config.n_clusters,
            0.5,
        )
        for p_index, p in enumerate(config.p_values):
            seed = distribution_seed + p_index + 1
            rng = np.random.default_rng(seed)
            distribution = SyntheticDistribution(base_distribution.means, p)
            X_train, y_train, X_test, y_test, X_pool, _ = _split_sample(
                distribution, rng, train_size, test_size, pool_size
            )
            S_test, S_pool = X_test[:, -1].astype(int), X_pool[:, -1].astype(int)
            model = RandomForestClassifier(random_state=seed)
            model.fit(X_train, y_train)
            probability_test = model.predict_proba(X_test)
            probability_pool = model.predict_proba(X_pool)

            started = time.perf_counter()
            unfair_prediction = model.predict(X_test)
            unfair_runtime = time.perf_counter() - started
            records.append(
                {
                    "Figure": 2,
                    "Repetition": repetition,
                    "Seed": seed,
                    "DistributionSeed": distribution_seed,
                    "p": p,
                    "Method": "Unfair",
                    "Epsilon": np.nan,
                    "Accuracy": accuracy_score(y_test, unfair_prediction),
                    "Unfairness": empirical_unfairness(
                        unfair_prediction, S_test, np.arange(config.n_classes)
                    ),
                    "RuntimeSeconds": unfair_runtime,
                }
            )

            for epsilon_index, epsilon in enumerate(config.epsilons):
                started = time.perf_counter()
                fair_prediction = postprocess_fn(
                    probability_test,
                    probability_pool,
                    S_test,
                    S_pool,
                    epsilon,
                    random_state=seed + 100 + epsilon_index,
                )
                runtime = time.perf_counter() - started
                records.append(
                    {
                        "Figure": 2,
                        "Repetition": repetition,
                        "Seed": seed,
                        "DistributionSeed": distribution_seed,
                        "p": p,
                        "Method": "Our Eps-Fair",
                        "Epsilon": epsilon,
                        "Accuracy": accuracy_score(y_test, fair_prediction),
                        "Unfairness": empirical_unfairness(
                            fair_prediction, S_test, np.arange(config.n_classes)
                        ),
                        "RuntimeSeconds": runtime,
                    }
                )
                if p == 0.75 and epsilon == 0.0:
                    for method, prediction in (
                        ("Unfair", unfair_prediction),
                        ("Exact fair", fair_prediction),
                    ):
                        for group in (-1, 1):
                            group_mask = S_test == group
                            for label in range(config.n_classes):
                                distribution_records.append(
                                    {
                                        "Figure": 3,
                                        "Repetition": repetition,
                                        "Seed": seed,
                                        "DistributionSeed": distribution_seed,
                                        "Method": method,
                                        "SensitiveGroup": group,
                                        "Class": label,
                                        "Probability": np.mean(
                                            prediction[group_mask] == label
                                        ),
                                    }
                                )

    raw = pd.DataFrame(records)
    summary = summarize_repetitions(
        raw, ["Figure", "p", "Method", "Epsilon"]
    )
    distribution_raw = pd.DataFrame(distribution_records)
    distribution_summary = (
        distribution_raw.groupby(
            ["Figure", "Method", "SensitiveGroup", "Class"], dropna=False
        )
        .agg(
            ProbabilityMean=("Probability", "mean"),
            ProbabilityStd=("Probability", "std"),
            Repetitions=("Repetition", "nunique"),
        )
        .reset_index()
    )
    _save_protocol_artifacts(config.output_dir, "figures2_3", raw, summary, config)
    distribution_raw.to_csv(
        Path(config.output_dir) / "figure3_distribution_raw.csv", index=False
    )
    distribution_summary.to_csv(
        Path(config.output_dir) / "figure3_distribution_summary.csv", index=False
    )
    return raw, summary, distribution_raw, distribution_summary


def run_synthetic_figure4(
    postprocess_fn: PostprocessFn, config=SyntheticProtocolConfig()
):
    """Compare independent 60/20/20 splitting to training-set calibration."""
    records = []
    for repetition in range(config.n_repetitions):
        seed = config.master_seed + 10000 * repetition + 400
        rng = np.random.default_rng(seed)
        distribution = make_synthetic_distribution(
            rng,
            config.n_classes,
            config.n_features,
            config.n_clusters,
            0.75,
        )
        complete, labels = distribution.sample(config.n_samples, rng)
        test_size = int(0.20 * config.n_samples)
        split_train_size = int(0.60 * config.n_samples)
        no_split_train_size = config.n_samples - test_size
        pool_start = split_train_size
        pool_stop = no_split_train_size
        test = complete[-test_size:]
        y_test = labels[-test_size:]
        S_test = test[:, -1].astype(int)

        for protocol, train, y_train, pool in (
            (
                "Split 60/20/20",
                complete[:split_train_size],
                labels[:split_train_size],
                complete[pool_start:pool_stop],
            ),
            (
                "No split 80/20",
                complete[:no_split_train_size],
                labels[:no_split_train_size],
                complete[:no_split_train_size],
            ),
        ):
            model = RandomForestClassifier(random_state=seed)
            model.fit(train, y_train)
            S_pool = pool[:, -1].astype(int)
            probability_test = model.predict_proba(test)
            probability_pool = model.predict_proba(pool)
            unfair_prediction = model.predict(test)
            fair_prediction = postprocess_fn(
                probability_test,
                probability_pool,
                S_test,
                S_pool,
                0.0,
                random_state=seed + (0 if protocol.startswith("Split") else 1),
            )
            for method, prediction in (
                ("Unfair", unfair_prediction),
                ("Exact fair", fair_prediction),
            ):
                records.append(
                    {
                        "Figure": 4,
                        "Repetition": repetition,
                        "Seed": seed,
                        "Protocol": protocol,
                        "Method": method,
                        "Accuracy": accuracy_score(y_test, prediction),
                        "Unfairness": empirical_unfairness(
                            prediction, S_test, np.arange(config.n_classes)
                        ),
                        "RuntimeSeconds": np.nan,
                    }
                )
    raw = pd.DataFrame(records)
    summary = summarize_repetitions(raw, ["Figure", "Protocol", "Method"])
    _save_protocol_artifacts(config.output_dir, "figure4", raw, summary, config)
    return raw, summary


def _postprocess_with_seed(
    postprocess_fn, probability_test, probability_pool, S_test, S_pool, epsilon, seed
):
    return postprocess_fn(
        probability_test,
        probability_pool,
        S_test,
        S_pool,
        epsilon,
        random_state=seed,
    )


def run_synthetic_figure10(
    postprocess_fn: PostprocessFn, config=SyntheticProtocolConfig()
):
    """Convergence against n and N using LightGBM, matching Figure 10 axes."""
    records = []
    classes = np.arange(config.n_classes)
    max_train = max(max(config.figure10_train_sizes), config.figure10_fixed_train_size)
    max_pool = max(config.figure10_fixed_pool_size, max(config.figure10_pool_sizes))

    for repetition in range(config.n_repetitions):
        seed = config.master_seed + 10000 * repetition + 1000
        rng = np.random.default_rng(seed)
        distribution = make_synthetic_distribution(
            rng,
            config.n_classes,
            config.n_features,
            config.n_clusters,
            0.75,
        )
        X_train_all, y_train_all = distribution.sample(max_train, rng)
        X_pool_all, _ = distribution.sample(max_pool, rng)
        X_test, y_test = distribution.sample(config.figure10_test_size, rng)
        X_oracle_pool, _ = distribution.sample(
            config.figure10_oracle_pool_size, rng
        )
        S_test = X_test[:, -1].astype(int)
        S_pool_all = X_pool_all[:, -1].astype(int)
        S_oracle = X_oracle_pool[:, -1].astype(int)
        true_probability_test = distribution.posterior(X_test)
        true_probability_pool = distribution.posterior(X_oracle_pool)

        oracle_predictions = {}
        for epsilon_index, epsilon in enumerate(config.figure10_epsilons):
            oracle_predictions[epsilon] = _postprocess_with_seed(
                postprocess_fn,
                true_probability_test,
                true_probability_pool,
                S_test,
                S_oracle,
                epsilon,
                seed + 7000 + epsilon_index,
            )
        bayes_prediction = np.argmax(true_probability_test, axis=1)

        fixed_pool = X_pool_all[: config.figure10_fixed_pool_size]
        fixed_S_pool = S_pool_all[: config.figure10_fixed_pool_size]
        for size_index, train_size in enumerate(config.figure10_train_sizes):
            model = LGBMClassifier(random_state=seed + size_index, verbose=-1)
            model.fit(X_train_all[:train_size], y_train_all[:train_size])
            probability_test = model.predict_proba(X_test)
            probability_pool = model.predict_proba(fixed_pool)
            unfair_prediction = np.argmax(probability_test, axis=1)
            records.append(
                {
                    "Figure": 10,
                    "Panel": "TrainingSize",
                    "Repetition": repetition,
                    "Seed": seed,
                    "Size": train_size,
                    "Method": "Unfair",
                    "Epsilon": np.nan,
                    "Accuracy": accuracy_score(y_test, unfair_prediction),
                    "Unfairness": empirical_unfairness(
                        unfair_prediction, S_test, classes
                    ),
                    "ReferenceAccuracy": accuracy_score(y_test, bayes_prediction),
                    "ReferenceUnfairness": empirical_unfairness(
                        bayes_prediction, S_test, classes
                    ),
                    "RuntimeSeconds": np.nan,
                }
            )
            for epsilon_index, epsilon in enumerate(config.figure10_epsilons):
                prediction = _postprocess_with_seed(
                    postprocess_fn,
                    probability_test,
                    probability_pool,
                    S_test,
                    fixed_S_pool,
                    epsilon,
                    seed + 100000 + 100 * size_index + epsilon_index,
                )
                reference = oracle_predictions[epsilon]
                records.append(
                    {
                        "Figure": 10,
                        "Panel": "TrainingSize",
                        "Repetition": repetition,
                        "Seed": seed,
                        "Size": train_size,
                        "Method": "Our Eps-Fair",
                        "Epsilon": epsilon,
                        "Accuracy": accuracy_score(y_test, prediction),
                        "Unfairness": empirical_unfairness(
                            prediction, S_test, classes
                        ),
                        "ReferenceAccuracy": accuracy_score(y_test, reference),
                        "ReferenceUnfairness": empirical_unfairness(
                            reference, S_test, classes
                        ),
                        "RuntimeSeconds": np.nan,
                    }
                )

        fixed_model = LGBMClassifier(random_state=seed + 5000, verbose=-1)
        fixed_model.fit(
            X_train_all[: config.figure10_fixed_train_size],
            y_train_all[: config.figure10_fixed_train_size],
        )
        fixed_probability_test = fixed_model.predict_proba(X_test)
        for pool_index, pool_size in enumerate(config.figure10_pool_sizes):
            probability_pool = fixed_model.predict_proba(X_pool_all[:pool_size])
            for epsilon_index, epsilon in enumerate(config.figure10_epsilons):
                prediction = _postprocess_with_seed(
                    postprocess_fn,
                    fixed_probability_test,
                    probability_pool,
                    S_test,
                    S_pool_all[:pool_size],
                    epsilon,
                    seed + 200000 + 100 * pool_index + epsilon_index,
                )
                reference = oracle_predictions[epsilon]
                records.append(
                    {
                        "Figure": 10,
                        "Panel": "PoolSize",
                        "Repetition": repetition,
                        "Seed": seed,
                        "Size": pool_size,
                        "Method": "Our Eps-Fair",
                        "Epsilon": epsilon,
                        "Accuracy": accuracy_score(y_test, prediction),
                        "Unfairness": empirical_unfairness(
                            prediction, S_test, classes
                        ),
                        "ReferenceAccuracy": accuracy_score(y_test, reference),
                        "ReferenceUnfairness": empirical_unfairness(
                            reference, S_test, classes
                        ),
                        "RuntimeSeconds": np.nan,
                    }
                )

    raw = pd.DataFrame(records)
    raw["ExcessRisk"] = (
        (1.0 - raw["Accuracy"]) - (1.0 - raw["ReferenceAccuracy"])
    )
    raw["UnfairnessDifference"] = (
        raw["Unfairness"] - raw["ReferenceUnfairness"]
    )
    summary = (
        raw.groupby(["Figure", "Panel", "Size", "Method", "Epsilon"], dropna=False)
        .agg(
            ExcessRiskMean=("ExcessRisk", "mean"),
            ExcessRiskStd=("ExcessRisk", "std"),
            UnfairnessDifferenceMean=("UnfairnessDifference", "mean"),
            UnfairnessDifferenceStd=("UnfairnessDifference", "std"),
            Repetitions=("Repetition", "nunique"),
        )
        .reset_index()
    )
    _save_protocol_artifacts(config.output_dir, "figure10", raw, summary, config)
    return raw, summary


@dataclass(frozen=True)
class BinaryProtocolConfig:
    n_repetitions: int = 30
    master_seed: int = 777
    epsilons: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    fairlearn_tolerances: tuple = (0.0001, 0.5, 1.0, 2.5, 5.0, 10.0)
    nn_epsilons: tuple = (0.01, 0.1, 0.3, 0.5, 0.9)
    adversarial_weights: tuple = (0.01, 0.1, 0.5, 0.9, 1.0)
    tuning_n_iter: int = 20
    cv_folds: int = 3
    run_adversarial: bool = True
    output_dir: str = "outputs/binary"


RF_DISTRIBUTIONS = {
    "n_estimators": np.arange(10, 201),
    "max_depth": np.arange(2, 17),
    "min_samples_split": np.arange(2, 11),
    "min_samples_leaf": np.arange(1, 9),
}

GBM_DISTRIBUTIONS = {
    "reg_alpha": [0, 0.1, 1, 2, 5, 10, 20, 50],
    "reg_lambda": [0, 0.1, 1, 2, 5, 10, 20, 50],
    "n_estimators": np.arange(10, 201),
    "num_leaves": np.arange(6, 51),
    "max_depth": np.arange(2, 17),
    "min_child_samples": np.arange(10, 101),
}


def _tuned_binary_models(X_train, y_train, seed, config):
    rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=seed),
        RF_DISTRIBUTIONS,
        n_iter=config.tuning_n_iter,
        cv=config.cv_folds,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)
    gbm = RandomizedSearchCV(
        LGBMClassifier(random_state=seed, verbose=-1),
        GBM_DISTRIBUTIONS,
        n_iter=config.tuning_n_iter,
        cv=config.cv_folds,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)
    return {
        "reglog": LogisticRegression(random_state=seed, max_iter=2000).fit(
            X_train, y_train
        ),
        "RF": rf.best_estimator_,
        "GBM": gbm.best_estimator_,
    }


def _binary_record(
    dataset,
    repetition,
    seed,
    model,
    method,
    prediction,
    y_test,
    S_test,
    runtime,
    epsilon=np.nan,
    fairlearn_tolerance=np.nan,
    adversarial_weight=np.nan,
):
    return {
        "Dataset": dataset,
        "K": 2,
        "Repetition": repetition,
        "Seed": seed,
        "Model": model,
        "Method": method,
        "Epsilon": epsilon,
        "FairlearnTolerance": fairlearn_tolerance,
        "AdversarialWeight": adversarial_weight,
        "Accuracy": accuracy_score(y_test, prediction),
        "Unfairness": empirical_unfairness(prediction, S_test, np.arange(2)),
        "RuntimeSeconds": runtime,
    }


def run_binary_real_protocol(
    dataset_name: str,
    loader: Callable,
    postprocess_fn: PostprocessFn,
    config=BinaryProtocolConfig(),
):
    """Figures 6-7 protocol with fresh splits/tuning across 30 repetitions."""
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    records = []
    for repetition in range(config.n_repetitions):
        seed = config.master_seed + 10000 * repetition + (
            0 if dataset_name.upper() == "DRUG" else 1000
        )
        X_train, X_test, y_train, y_test, X_pool = loader(K=2, seed=seed)
        S_train = X_train[:, -1].astype(int)
        S_test = X_test[:, -1].astype(int)
        S_pool = X_pool[:, -1].astype(int)
        models = _tuned_binary_models(X_train, y_train, seed, config)

        for model_index, (model_name, model) in enumerate(models.items()):
            probability_test = model.predict_proba(X_test)
            probability_pool = model.predict_proba(X_pool)
            started = time.perf_counter()
            prediction = model.predict(X_test)
            records.append(
                _binary_record(
                    dataset_name,
                    repetition,
                    seed,
                    model_name,
                    "Unfair",
                    prediction,
                    y_test,
                    S_test,
                    time.perf_counter() - started,
                )
            )

            for epsilon_index, epsilon in enumerate(config.epsilons):
                started = time.perf_counter()
                prediction = postprocess_fn(
                    probability_test,
                    probability_pool,
                    S_test,
                    S_pool,
                    epsilon,
                    random_state=seed + 100 * model_index + epsilon_index,
                )
                records.append(
                    _binary_record(
                        dataset_name,
                        repetition,
                        seed,
                        model_name,
                        "Our Eps-Fair",
                        prediction,
                        y_test,
                        S_test,
                        time.perf_counter() - started,
                        epsilon=epsilon,
                    )
                )

            for tolerance_index, tolerance in enumerate(
                config.fairlearn_tolerances
            ):
                # Paper grid maps to ExponentiatedGradient.eps. Values above 1
                # cannot represent a DP difference_bound, whose range is [0, 1].
                mitigator = ExponentiatedGradient(
                    estimator=clone(model),
                    constraints=DemographicParity(),
                    eps=tolerance,
                    max_iter=50,
                )
                started = time.perf_counter()
                mitigator.fit(
                    X_train, y_train, sensitive_features=S_train
                )
                prediction = mitigator.predict(
                    X_test, random_state=seed + 500 + tolerance_index
                )
                records.append(
                    _binary_record(
                        dataset_name,
                        repetition,
                        seed,
                        model_name,
                        "Fairlearn",
                        prediction,
                        y_test,
                        S_test,
                        time.perf_counter() - started,
                        fairlearn_tolerance=tolerance,
                    )
                )

        if config.run_adversarial:
            records.extend(
                _run_binary_neural_baselines(
                    dataset_name,
                    repetition,
                    seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    X_pool,
                    S_train,
                    S_test,
                    S_pool,
                    postprocess_fn,
                    config,
                )
            )

    raw = pd.DataFrame(records)
    keys = [
        "Dataset",
        "K",
        "Model",
        "Method",
        "Epsilon",
        "FairlearnTolerance",
        "AdversarialWeight",
    ]
    summary = summarize_repetitions(raw, keys)
    prefix = f"{dataset_name.lower()}_figures6_7"
    _save_protocol_artifacts(config.output_dir, prefix, raw, summary, config)
    return raw, summary


def _run_binary_neural_baselines(
    dataset_name,
    repetition,
    seed,
    X_train,
    X_test,
    y_train,
    y_test,
    X_pool,
    S_train,
    S_test,
    S_pool,
    postprocess_fn,
    config,
):
    import tensorflow.compat.v1 as tf
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.datasets import BinaryLabelDataset
    from sklearn.neural_network import MLPClassifier

    def as_aif360(X, y, sensitive):
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

    records = []
    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        batch_size=128,
        max_iter=200,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    probability_test = model.predict_proba(X_test)
    probability_pool = model.predict_proba(X_pool)
    prediction = model.predict(X_test)
    records.append(
        _binary_record(
            dataset_name,
            repetition,
            seed,
            "NN",
            "Unfair",
            prediction,
            y_test,
            S_test,
            np.nan,
        )
    )
    for epsilon_index, epsilon in enumerate(config.nn_epsilons):
        started = time.perf_counter()
        prediction = postprocess_fn(
            probability_test,
            probability_pool,
            S_test,
            S_pool,
            epsilon,
            random_state=seed + 800 + epsilon_index,
        )
        records.append(
            _binary_record(
                dataset_name,
                repetition,
                seed,
                "NN",
                "Our Eps-Fair",
                prediction,
                y_test,
                S_test,
                time.perf_counter() - started,
                epsilon=epsilon,
            )
        )

    tf.disable_eager_execution()
    training = as_aif360(X_train, y_train, S_train)
    testing = as_aif360(X_test, y_test, S_test)
    for weight_index, weight in enumerate(config.adversarial_weights):
        tf.reset_default_graph()
        tf.set_random_seed(seed + weight_index)
        session = tf.Session()
        try:
            adversarial = AdversarialDebiasing(
                privileged_groups=[{"sensitive_attr": 1}],
                unprivileged_groups=[{"sensitive_attr": 0}],
                scope_name=f"adv_{dataset_name}_{repetition}_{weight_index}",
                debias=True,
                sess=session,
                num_epochs=200,
                batch_size=128,
                classifier_num_hidden_units=50,
                adversary_loss_weight=weight,
            )
            started = time.perf_counter()
            adversarial.fit(training)
            prediction = adversarial.predict(testing).labels.ravel()
            runtime = time.perf_counter() - started
        finally:
            session.close()
        records.append(
            _binary_record(
                dataset_name,
                repetition,
                seed,
                "NN",
                "Fair-adversarial",
                prediction,
                y_test,
                S_test,
                runtime,
                adversarial_weight=weight,
            )
        )
    return records


def plot_synthetic_figure1(summary, save_path=None):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(
        summary["p"],
        summary["UnfairnessMean"],
        yerr=summary["UnfairnessStd"],
        marker="o",
        capsize=3,
    )
    axis.set(xlabel="p", ylabel="Bayes classifier unfairness", title="Figure 1")
    axis.grid(alpha=0.3)
    if save_path:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure


def plot_synthetic_figure4(raw, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=raw, x="Protocol", y="Unfairness", hue="Method", ax=axes[0])
    sns.boxplot(data=raw, x="Protocol", y="Accuracy", hue="Method", ax=axes[1])
    axes[0].set_title("Unfairness - lower is better")
    axes[1].set_title("Accuracy - higher is better")
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure


def plot_synthetic_figures2_3(
    summary, distribution_summary, figure2_path=None, figure3_path=None
):
    """Plot phase diagrams and aggregated prediction distributions."""
    import matplotlib.pyplot as plt

    figure2, axes = plt.subplots(1, 2, figsize=(14, 5))
    left_p_values = (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
    for epsilon in (0.0, 0.05, 0.1):
        values = summary[
            (summary["Method"] == "Our Eps-Fair")
            & (summary["Epsilon"] == epsilon)
            & (summary["p"].isin(left_p_values))
        ].sort_values("p")
        axes[0].errorbar(
            values["UnfairnessMean"],
            values["AccuracyMean"],
            xerr=values["UnfairnessStd"],
            yerr=values["AccuracyStd"],
            marker="o",
            capsize=3,
            label=f"epsilon={epsilon:g}",
        )
    unfair = summary[
        (summary["Method"] == "Unfair") & (summary["p"].isin(left_p_values))
    ].sort_values("p")
    axes[0].errorbar(
        unfair["UnfairnessMean"],
        unfair["AccuracyMean"],
        xerr=unfair["UnfairnessStd"],
        yerr=unfair["AccuracyStd"],
        marker="^",
        capsize=3,
        label="unfair",
    )

    p75 = summary[summary["p"] == 0.75]
    fair = p75[p75["Method"] == "Our Eps-Fair"].sort_values("Epsilon")
    unfair = p75[p75["Method"] == "Unfair"]
    axes[1].errorbar(
        fair["UnfairnessMean"],
        fair["AccuracyMean"],
        xerr=fair["UnfairnessStd"],
        yerr=fair["AccuracyStd"],
        marker="o",
        capsize=3,
        label="epsilon-fair",
    )
    axes[1].errorbar(
        unfair["UnfairnessMean"],
        unfair["AccuracyMean"],
        xerr=unfair["UnfairnessStd"],
        yerr=unfair["AccuracyStd"],
        marker="^",
        capsize=3,
        linestyle="none",
        label="unfair",
    )
    axes[0].set_title("Figure 2-left: varying p")
    axes[1].set_title("Figure 2-right: p=0.75")
    for axis in axes:
        axis.set(xlabel="Unfairness", ylabel="Accuracy")
        axis.grid(alpha=0.3)
        axis.legend()
    figure2.tight_layout()
    if figure2_path:
        figure2.savefig(figure2_path, dpi=200, bbox_inches="tight")

    methods = ("Unfair", "Exact fair")
    figure3, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    width = 0.35
    for axis, method in zip(axes, methods):
        subset = distribution_summary[
            distribution_summary["Method"] == method
        ]
        for group_index, group in enumerate((-1, 1)):
            values = subset[subset["SensitiveGroup"] == group].sort_values(
                "Class"
            )
            offset = (-0.5 if group_index == 0 else 0.5) * width
            axis.bar(
                values["Class"] + offset,
                values["ProbabilityMean"],
                width=width,
                yerr=values["ProbabilityStd"],
                capsize=3,
                label=f"S={group}",
            )
        axis.set(
            xlabel="Class",
            ylabel="Prediction probability",
            title=f"Figure 3: {method}",
        )
        axis.legend()
    figure3.tight_layout()
    if figure3_path:
        figure3.savefig(figure3_path, dpi=200, bbox_inches="tight")
    return figure2, figure3


def plot_synthetic_figure10(summary, save_path=None):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    left = summary[summary["Panel"] == "TrainingSize"]
    right = summary[summary["Panel"] == "PoolSize"]
    for (method, epsilon), values in left.groupby(
        ["Method", "Epsilon"], dropna=False
    ):
        label = "unfair" if method == "Unfair" else f"epsilon={epsilon:g}"
        axes[0].errorbar(
            values["Size"],
            values["ExcessRiskMean"],
            yerr=values["ExcessRiskStd"],
            label=label,
            capsize=2,
        )
    for epsilon, values in right.groupby("Epsilon"):
        axes[1].errorbar(
            values["Size"],
            values["UnfairnessDifferenceMean"],
            yerr=values["UnfairnessDifferenceStd"],
            label=f"epsilon={epsilon:g}",
            capsize=2,
        )
    axes[0].set(xlabel="n: training samples", ylabel="Excess risk")
    axes[1].set(xlabel="N: unlabeled samples", ylabel="Unfairness difference")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure


def plot_binary_repeated(summary, dataset_name, figure, save_path=None):
    """Plot Figure 6 or 7 from 30-run mean/std summaries."""
    import matplotlib.pyplot as plt

    if figure == 6:
        data = summary[summary["Model"].isin(["reglog", "RF", "GBM"])]
        models = ("reglog", "RF", "GBM")
        methods = ("Our Eps-Fair", "Fairlearn")
    elif figure == 7:
        data = summary[summary["Model"] == "NN"]
        models = ("NN",)
        methods = ("Our Eps-Fair", "Fair-adversarial")
    else:
        raise ValueError("figure must be 6 or 7.")

    figure_object, axes = plt.subplots(
        1, len(models), figsize=(6 * len(models), 5), squeeze=False
    )
    for axis, model in zip(axes.ravel(), models):
        subset = data[data["Model"] == model]
        for method in methods:
            values = subset[subset["Method"] == method].sort_values(
                "UnfairnessMean"
            )
            axis.errorbar(
                values["UnfairnessMean"],
                values["AccuracyMean"],
                xerr=values["UnfairnessStd"],
                yerr=values["AccuracyStd"],
                marker="o",
                capsize=3,
                label=method,
            )
        unfair = subset[subset["Method"] == "Unfair"]
        axis.errorbar(
            unfair["UnfairnessMean"],
            unfair["AccuracyMean"],
            xerr=unfair["UnfairnessStd"],
            yerr=unfair["AccuracyStd"],
            marker="^",
            linestyle="none",
            label="Unfair",
        )
        axis.set(
            xlabel="Unfairness",
            ylabel="Accuracy",
            title=f"{dataset_name} - {model}",
        )
        axis.grid(alpha=0.3)
        axis.legend()
    figure_object.tight_layout()
    if save_path:
        figure_object.savefig(save_path, dpi=200, bbox_inches="tight")
    return figure_object
