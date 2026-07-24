"""Run the documented UCI DRUG/CRIME group-adaptation pilot.

This runner is intentionally explicit about the CRIME adaptation profile.  It is
not a paper-exact recreation of the unpublished Crime.mat/Scat preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from scipy.stats import randint
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fair-projection/enem/multi-group-multi-class"))
from GroupFair import GFair  # noqa: E402


DRUG_COLUMNS = [
    "id", "age", "gender", "education", "country", "ethnicity", "nscore",
    "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "alcohol",
    "amphet", "amyl", "benzos", "caff", "cannabis", "choc", "coke", "crack",
    "ecstasy", "heroin", "ketamine", "legalh", "lsd", "meth", "mushrooms",
    "nicotine", "semer", "vsa",
]
DRUG_EDUCATION_DEGREE = {0.45468, 1.16365, 1.98437}


def parse_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        content = raw.split("#", 1)[0].strip()
        if not content or "=" not in content:
            continue
        key, value = (item.strip() for item in content.split("=", 1))
        result[key] = value
    return result


def csv_numbers(value: str, kind=float) -> list:
    return [kind(item) for item in value.split(",")]


def expand_seeds(value: str) -> list[int]:
    if ":" in value:
        start, end = (int(item) for item in value.split(":"))
        return list(range(start, end + 1))
    return csv_numbers(value, int)


def split_indices(y: np.ndarray, sensitive: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create disjoint 60/20/20 splits stratified by (Y,S), with Y fallback."""
    indices = np.arange(len(y))
    joint = np.asarray([f"{label}|{group}" for label, group in zip(y, sensitive)])
    warning = None
    try:
        train, remaining = train_test_split(indices, train_size=0.6, random_state=seed, shuffle=True, stratify=joint)
        calibration, test = train_test_split(remaining, train_size=0.5, random_state=seed + 10_000, shuffle=True, stratify=joint[remaining])
        strategy = "joint_y_sensitive"
    except ValueError as exc:
        warning = f"joint (Y,S) stratification unavailable: {exc}; fell back to Y"
        train, remaining = train_test_split(indices, train_size=0.6, random_state=seed, shuffle=True, stratify=y)
        calibration, test = train_test_split(remaining, train_size=0.5, random_state=seed + 10_000, shuffle=True, stratify=y[remaining])
        strategy = "y_fallback"
    detail = {"strategy": strategy, "warning": warning, "joint_counts": {str(key): int((joint == key).sum()) for key in np.unique(joint)}}
    return np.sort(train), np.sort(calibration), np.sort(test), detail


def max_pairwise_dp(y_pred: np.ndarray, sensitive: np.ndarray, n_classes: int) -> float:
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return float("nan")
    rates = []
    for group in groups:
        mask = sensitive == group
        if not mask.any():
            return float("nan")
        rates.append([(y_pred[mask] == klass).mean() for klass in range(n_classes)])
    return float(np.max(np.ptp(np.asarray(rates), axis=0)))


def normalized_probability(value: np.ndarray) -> np.ndarray:
    """Normalize the FairProjection core output from (N, K, 1) to (N, K)."""
    probability = np.asarray(value)
    if probability.ndim == 3 and probability.shape[-1] == 1:
        probability = probability[..., 0]
    if probability.ndim != 2:
        raise ValueError(f"Expected probability shape (N, K), got {probability.shape}")
    return probability

def read_crime() -> pd.DataFrame:
    names_path = ROOT / "data/CRIME/raw/uci/communities.names"
    names = []
    for line in names_path.read_text(encoding="latin-1").splitlines():
        if line.startswith("@attribute "):
            names.append(line.split()[1])
    if len(names) != 128:
        raise ValueError(f"Expected 128 CRIME columns, found {len(names)}")
    return pd.read_csv(ROOT / "data/CRIME/raw/uci/communities.data", names=names, na_values="?")


def prepare_drug(seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    raw = pd.read_csv(ROOT / "data/DRUG/raw/uci/drug_consumption.data", header=None, names=DRUG_COLUMNS)
    label_map = {"CL0": 0, "CL1": 1, "CL2": 1, "CL3": 2, "CL4": 2, "CL5": 2, "CL6": 3}
    y = raw["cannabis"].map(label_map).to_numpy(dtype=int)
    sensitive = raw["education"].isin(DRUG_EDUCATION_DEGREE).astype(int).to_numpy()
    feature_columns = DRUG_COLUMNS[1:13]
    feature_columns.remove("education")
    X = raw.loc[:, feature_columns].astype(float)
    metadata = {
        "profile": "group-adaptation: strict degree; professional certificate/diploma is non-degree",
        "label": "Cannabis: CL0/CL1-2/CL3-5/CL6 -> 0/1/2/3",
        "sensitive": "education in {University, Masters, Doctorate}",
        "feature_columns": feature_columns,
    }
    return X, y, sensitive, metadata


def prepare_crime(seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    raw = read_crime()
    target = raw["ViolentCrimesPerPop"].astype(float).to_numpy()
    edges = np.quantile(target, np.linspace(0, 1, 6))
    if len(np.unique(edges)) != 6:
        raise ValueError("CRIME train quantiles are not unique; cannot create K=5")
    y = np.digitize(target, edges[1:-1], right=True).astype(int)
    race = raw["racepctblack"].astype(float).to_numpy()
    threshold = float(np.quantile(race, 0.70))
    sensitive = (race >= threshold).astype(int)
    drop = {"state", "county", "community", "communityname", "fold", "ViolentCrimesPerPop", "racepctblack"}
    X = raw.drop(columns=list(drop)).apply(pd.to_numeric, errors="coerce")
    missing_rate = X.isna().mean()
    dropped_missing = missing_rate[missing_rate > 0.50].index.tolist()
    X = X.drop(columns=dropped_missing)
    metadata = {
        "profile": "group-adaptation; not paper-exact because Crime.mat/Scat is unpublished",
        "label": "ViolentCrimesPerPop, full-cohort quintiles K=5",
        "quantile_policy": "Full-cohort target quantiles are used only to define the adaptation label before stratified splitting.",
        "bin_edges": edges.tolist(),
        "sensitive": "racepctblack >= full-cohort 0.70 quantile",
        "missing_policy": "Median imputation is fit on the training split; columns with cohort missingness >50% are removed before splitting.",
        "sensitive_threshold": threshold,
        "dropped_identifiers": sorted(drop),
        "dropped_missing_gt_50pct": dropped_missing,
        "missing_rate_cohort": missing_rate.to_dict(),
    }
    return X, y, sensitive, metadata


def split_and_transform(X: pd.DataFrame, y: np.ndarray, sensitive: np.ndarray, seed: int):
    train_idx, calibration_idx, test_idx, split_detail = split_indices(y, sensitive, seed)
    if any(len(np.unique(y[index])) != len(np.unique(y)) for index in (train_idx, calibration_idx, test_idx)):
        raise ValueError("A split is missing a class; choose another seed or revise split protocol")
    if any(len(np.unique(sensitive[index])) != 2 for index in (train_idx, calibration_idx, test_idx)):
        raise ValueError("A split is missing a sensitive group; choose another seed or revise threshold")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X.iloc[train_idx])
    X_calibration = imputer.transform(X.iloc[calibration_idx])
    X_test = imputer.transform(X.iloc[test_idx])
    return (train_idx, calibration_idx, test_idx), (X_train, X_calibration, X_test), imputer, split_detail


def tuning_distributions(config: dict[str, str]):
    def bounded(name: str):
        lo, hi = (int(item) for item in config[name].split(":"))
        return randint(lo, hi + 1)
    return {
        "reg_alpha": csv_numbers(config["tuning.reg_alpha"]),
        "reg_lambda": csv_numbers(config["tuning.reg_lambda"]),
        "n_estimators": bounded("tuning.n_estimators"),
        "num_leaves": bounded("tuning.num_leaves"),
        "max_depth": bounded("tuning.max_depth"),
        "min_child_samples": bounded("tuning.min_child_samples"),
    }


def run_dataset(dataset: str, seed: int, config: dict[str, str], out: Path, n_iter: int, max_iter: int) -> list[dict]:
    X, y, sensitive, metadata = prepare_drug(seed) if dataset == "DRUG" else prepare_crime(seed)
    (train_idx, calibration_idx, test_idx), (X_train, X_cal, X_test), imputer, split_detail = split_and_transform(X, y, sensitive, seed)
    n_classes = int(y.max() + 1)
    ds_dir = out / "preprocessing"
    ds_dir.mkdir(parents=True, exist_ok=True)
    metadata.update({
        "dataset": dataset, "seed": seed, "n_rows": len(y), "n_features": X_train.shape[1],
        "class_counts": np.bincount(y, minlength=n_classes).tolist(),
        "sensitive_counts": np.bincount(sensitive, minlength=2).tolist(),
        "split": split_detail,
        "split_counts": {name: {"y": np.bincount(y[index], minlength=n_classes).tolist(), "sensitive": np.bincount(sensitive[index], minlength=2).tolist()} for name, index in {"train": train_idx, "calibration": calibration_idx, "test": test_idx}.items()},
        "imputer_medians": dict(zip(X.columns, imputer.statistics_.tolist())),
    })
    (ds_dir / f"{dataset.lower()}_seed-{seed}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    split_dir = out / "splits"; split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(split_dir / f"{dataset}_seed-{seed}.npz", train=train_idx, calibration=calibration_idx, test=test_idx)

    cv = StratifiedKFold(n_splits=int(config["tuning.cv"]), shuffle=config["tuning.stratified"].lower() == "true", random_state=seed)
    base = LGBMClassifier(random_state=seed, verbosity=-1)
    search = RandomizedSearchCV(base, tuning_distributions(config), n_iter=n_iter, scoring=config["tuning.scoring"], cv=cv, random_state=seed, n_jobs=1, refit=True, error_score="raise")
    print(f"[{dataset} seed={seed}] LightGBM tuning: {n_iter} candidates x {config['tuning.cv']} folds", flush=True)
    started = time.perf_counter(); search.fit(X_train, y[train_idx]); tuning_s = time.perf_counter() - started
    tuning_dir = out / "tuning"; tuning_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(tuning_dir / f"{dataset}_seed-{seed}_cv_results.csv", index=False)
    (tuning_dir / f"{dataset}_seed-{seed}_best_params.json").write_text(json.dumps({"best_params": search.best_params_, "best_score": search.best_score_, "tuning_runtime_s": tuning_s}, indent=2), encoding="utf-8")
    model_dir = out / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    search.best_estimator_.booster_.save_model(str(model_dir / f"{dataset}_seed-{seed}_lightgbm.txt"))
    pred_dir = out / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    base_calibration_probability = search.predict_proba(X_cal)
    base_test_probability = search.predict_proba(X_test)
    np.savez(pred_dir / f"{dataset}_seed-{seed}_base_calibration.npz", probability=base_calibration_probability, y=y[calibration_idx], sensitive=sensitive[calibration_idx])
    np.savez(pred_dir / f"{dataset}_seed-{seed}_base_test.npz", probability=base_test_probability, y=y[test_idx], sensitive=sensitive[test_idx])

    print(f"[{dataset} seed={seed}] tuning complete in {tuning_s:.1f}s; starting FairProjection", flush=True)
    gf = GFair(search.best_estimator_, clf_S=None, clf_SgY=None, div=config["fairprojection.divergence"])
    gf.fit(X_train, y[train_idx], sensitive[train_idx], sample_weight=None)
    base_test_prediction = base_test_probability.argmax(axis=1)
    base_calibration_prediction = base_calibration_probability.argmax(axis=1)
    rows = [{
        "dataset": dataset, "seed": seed, "method": "Unconstrained", "alpha": np.nan,
        "divergence": "none", "status": "success", "completion_status": "completed_without_exception",
        "convergence": "not_applicable", "accuracy": accuracy_score(y[test_idx], base_test_prediction),
        "macro_f1": f1_score(y[test_idx], base_test_prediction, average="macro"),
        "dp_unfairness": max_pairwise_dp(base_test_prediction, sensitive[test_idx], n_classes),
        "calibration_dp_unfairness": max_pairwise_dp(base_calibration_prediction, sensitive[calibration_idx], n_classes),
        "empirical_dp_violation": np.nan, "calibration_empirical_dp_violation": np.nan,
        "solver_residual": np.nan, "tuning_runtime_s": tuning_s, "projection_runtime_s": 0.0,
        "total_runtime_s": tuning_s, "note": "Predictions from the shared LightGBM base probabilities."
    }]
    log_dir = out / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    for alpha_index, alpha in enumerate(csv_numbers(config["fairprojection.alpha"])):
        # Constraint construction is deterministic; retain this identifier only for provenance.
        print(f"[{dataset} seed={seed} alpha={alpha:g}] projection start (max_iter={max_iter})", flush=True)
        started = time.perf_counter()
        row = {"dataset": dataset, "seed": seed, "method": "FairProjection", "alpha": alpha, "divergence": config["fairprojection.divergence"], "status": "failed", "completion_status": "not_completed", "convergence": "convergence_not_observable", "tuning_runtime_s": tuning_s}
        try:
            gf.project(X_cal, s=sensitive[calibration_idx], constraints=[("sp", alpha)], rho=float(config["fairprojection.rho"]), max_iter=max_iter, method=config["fairprojection.projection_method"], verbose=True)
            test_probability = normalized_probability(gf.predict_proba(X_test, s=sensitive[test_idx]))
            calibration_probability = normalized_probability(gf.predict_proba(X_cal, s=sensitive[calibration_idx]))
            test_prediction = test_probability.argmax(axis=1)
            cal_prediction = calibration_probability.argmax(axis=1)
            projection_s = time.perf_counter() - started
            np.savez(pred_dir / f"{dataset}_seed-{seed}_alpha-{alpha}_projected_test.npz", probability=test_probability, prediction=test_prediction, y=y[test_idx], sensitive=sensitive[test_idx])
            np.savez(pred_dir / f"{dataset}_seed-{seed}_alpha-{alpha}_projected_calibration.npz", probability=calibration_probability, prediction=cal_prediction, y=y[calibration_idx], sensitive=sensitive[calibration_idx])
            print(f"[{dataset} seed={seed} alpha={alpha:g}] projection complete in {projection_s:.1f}s", flush=True)
            row.update({
                "status": "success", "completion_status": "completed_without_exception", "convergence": "convergence_not_observable", "accuracy": accuracy_score(y[test_idx], test_prediction),
                "macro_f1": f1_score(y[test_idx], test_prediction, average="macro"),
                "dp_unfairness": max_pairwise_dp(test_prediction, sensitive[test_idx], n_classes),
                "calibration_dp_unfairness": max_pairwise_dp(cal_prediction, sensitive[calibration_idx], n_classes),
                "solver_residual": np.nan, "empirical_dp_violation": max(0.0, max_pairwise_dp(test_prediction, sensitive[test_idx], n_classes) - alpha), "calibration_empirical_dp_violation": max(0.0, max_pairwise_dp(cal_prediction, sensitive[calibration_idx], n_classes) - alpha), "projection_runtime_s": projection_s,
                "total_runtime_s": tuning_s + projection_s,
                "note": "Core does not expose solver diagnostics; this is completed_without_exception, not a convergence claim.",
            })
        except Exception as exc:  # solver failures are cells, not orchestration failures
            row.update({"error": repr(exc), "projection_runtime_s": time.perf_counter() - started})
            log_dir = out / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{dataset}_seed-{seed}_alpha-{alpha}_ce.error.txt").write_text(repr(exc), encoding="utf-8")
        (log_dir / f"{dataset}_seed-{seed}_alpha-{alpha}_ce.json").write_text(json.dumps({"status": row["status"], "projection_seed": None, "max_iter_requested": max_iter, "rho": float(config["fairprojection.rho"]), "convergence": row.get("convergence"), "projection_runtime_s": row.get("projection_runtime_s"), "error": row.get("error")}, indent=2), encoding="utf-8")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproduction.default.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", default="DRUG,CRIME")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-iter", type=int, default=None, help="Override proposed config value for a smoke run.")
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    config = parse_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps({"config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "profile": "UCI group-adaptation"}, indent=2), encoding="utf-8")
    rows = []
    n_iter = args.n_iter if args.n_iter is not None else int(config["tuning.n_iter"])
    for dataset in [item.strip().upper() for item in args.datasets.split(",")]:
        for seed in expand_seeds(args.seeds):
            rows.extend(run_dataset(dataset, seed, config, args.out, n_iter, args.max_iter))
            pd.DataFrame(rows).to_csv(args.out / "metrics.csv", index=False)
    plot = pd.DataFrame(rows); plot = plot[plot["status"] == "success"]
    if not plot.empty:
        fig, axis = plt.subplots(figsize=(7, 5))
        for dataset, part in plot.groupby("dataset"):
            baseline = part[part["method"] == "Unconstrained"]
            projected = part[part["method"] == "FairProjection"]
            if not projected.empty:
                summary = projected.groupby("alpha", as_index=False)[["accuracy", "dp_unfairness"]].mean().sort_values("alpha")
                axis.plot(summary["dp_unfairness"], summary["accuracy"], marker="o", label=dataset)
            if not baseline.empty:
                point = baseline[["accuracy", "dp_unfairness"]].mean()
                axis.scatter(point["dp_unfairness"], point["accuracy"], marker="x", s=55, label=f"{dataset} Unconstrained")
        axis.set(xlabel="DP unfairness (thấp hơn tốt hơn)", ylabel="Accuracy test (cao hơn tốt hơn)", title="FairProjection + LightGBM: accuracy–unfairness")
        axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
        fig.savefig(args.out / "accuracy_unfairness_curve.png", dpi=160); plt.close(fig)
    failures = [row for row in rows if row["status"] == "failed"]
    pd.DataFrame(failures, columns=pd.DataFrame(rows).columns).to_csv(args.out / "failures.csv", index=False)


if __name__ == "__main__":
    main()
