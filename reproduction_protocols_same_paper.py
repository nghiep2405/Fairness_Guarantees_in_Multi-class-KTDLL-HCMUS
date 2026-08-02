"""Redraw the report figures from existing experiment summaries.

This module is intentionally read-only with respect to ``outputs/``.  It
loads the CSV artifacts already produced by the experiment protocols and
writes newly rendered figures to ``figures_same_paper/`` (or to a directory
selected with ``--output-dir``).

Unlike the plotting helpers in ``reproduction_protocols.py``, these plots do
not draw standard-deviation error bars.  The data components are therefore
shown as ordinary markers instead of marker-plus-error-bar crosses.

Run all supported figures (Figure 4 is deliberately excluded):

    python reproduction_protocols_same_paper.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Render directly to PNG without depending on Tk/Qt GUI installations.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "outputs"
DEFAULT_FIGURE_DIR = PROJECT_DIR / "figures_same_paper"


def _paper_style() -> None:
    """Apply a compact, paper-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finish_axis(axis) -> None:
    axis.grid(True, linestyle="--", alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False)


def _save(figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing experiment summary: {path}\n"
            "Run the corresponding experiment first or select another "
            "--data-dir."
        )
    return pd.read_csv(path)


def plot_synthetic_figure1(summary, save_path=None):
    """Plot Figure 1 with ordinary circular markers and no error bars."""
    values = summary.sort_values("p")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        values["p"],
        values["UnfairnessMean"],
        color="tab:blue",
        marker="o",
    )
    axis.set(
        xlabel="p",
        ylabel="Bayes classifier unfairness",
        title="Figure 1",
    )
    _finish_axis(axis)
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def plot_synthetic_figures2_3(
    summary, distribution_summary, figure2_path=None, figure3_path=None
):
    """Plot Figures 2 and 3 without standard-deviation error bars."""
    figure2, axes = plt.subplots(1, 2, figsize=(14, 5))
    left_p_values = (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
    curve_styles = {
        0.0: ("o", "tab:blue"),
        0.05: ("v", "tab:green"),
        0.1: ("s", "tab:red"),
    }

    for epsilon, (marker, color) in curve_styles.items():
        values = summary[
            (summary["Method"] == "Our Eps-Fair")
            & np.isclose(summary["Epsilon"], epsilon, equal_nan=False)
            & summary["p"].isin(left_p_values)
        ].sort_values("p")
        axes[0].plot(
            values["UnfairnessMean"],
            values["AccuracyMean"],
            marker=marker,
            color=color,
            label=rf"$\epsilon={epsilon:g}$",
        )

    unfair = summary[
        (summary["Method"] == "Unfair")
        & summary["p"].isin(left_p_values)
    ].sort_values("p")
    axes[0].plot(
        unfair["UnfairnessMean"],
        unfair["AccuracyMean"],
        marker="^",
        color="tab:orange",
        label="Unfair",
    )

    p75 = summary[np.isclose(summary["p"], 0.75)]
    fair = p75[p75["Method"] == "Our Eps-Fair"].sort_values("Epsilon")
    unfair = p75[p75["Method"] == "Unfair"]
    axes[1].plot(
        fair["UnfairnessMean"],
        fair["AccuracyMean"],
        marker="o",
        color="tab:red",
        label=r"$\epsilon$-fair",
    )
    axes[1].scatter(
        unfair["UnfairnessMean"],
        unfair["AccuracyMean"],
        marker="^",
        s=70,
        color="tab:orange",
        label="Unfair",
        zorder=3,
    )

    axes[0].set_title("Figure 2-left: varying p")
    axes[1].set_title("Figure 2-right: p=0.75")
    for axis in axes:
        axis.set(xlabel="Unfairness", ylabel="Accuracy")
        _finish_axis(axis)
    figure2.tight_layout()
    _save(figure2, figure2_path)

    methods = ("Unfair", "Exact fair")
    figure3, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    width = 0.35
    for axis, method in zip(axes, methods):
        subset = distribution_summary[
            distribution_summary["Method"] == method
        ]
        for group_index, group in enumerate((-1, 1)):
            values = subset[
                subset["SensitiveGroup"] == group
            ].sort_values("Class")
            offset = (-0.5 if group_index == 0 else 0.5) * width
            axis.bar(
                values["Class"] + offset,
                values["ProbabilityMean"],
                width=width,
                label=f"S={group}",
                color=("tab:blue" if group == -1 else "tab:red"),
                alpha=0.85,
            )
        axis.set(
            xlabel="Class",
            ylabel="Prediction probability",
            title=f"Figure 3: {method}",
        )
        axis.legend(frameon=False)
    figure3.tight_layout()
    _save(figure3, figure3_path)
    return figure2, figure3


def plot_synthetic_figure10(summary, save_path=None):
    """Plot Figure 10 with normal markers and no error-bar crosses."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    left = summary[summary["Panel"] == "TrainingSize"]
    right = summary[summary["Panel"] == "PoolSize"]
    markers = ("o", "v", "s", "D", "P", "X")

    unfair = left[left["Method"] == "Unfair"].sort_values("Size")
    if not unfair.empty:
        axes[0].plot(
            unfair["Size"],
            unfair["ExcessRiskMean"],
            marker="^",
            color="tab:orange",
            label="Unfair",
        )

    fair_left = left[left["Method"] == "Our Eps-Fair"]
    for index, (epsilon, values) in enumerate(
        fair_left.groupby("Epsilon", sort=True)
    ):
        values = values.sort_values("Size")
        axes[0].plot(
            values["Size"],
            values["ExcessRiskMean"],
            marker=markers[index % len(markers)],
            label=rf"$\epsilon={epsilon:g}$",
        )

    for index, (epsilon, values) in enumerate(
        right.groupby("Epsilon", sort=True)
    ):
        values = values.sort_values("Size")
        axes[1].plot(
            values["Size"],
            values["UnfairnessDifferenceMean"],
            marker=markers[index % len(markers)],
            label=rf"$\epsilon={epsilon:g}$",
        )

    axes[0].set(xlabel="n: training samples", ylabel="Excess risk")
    axes[1].set(
        xlabel="N: unlabeled samples", ylabel="Unfairness difference"
    )
    for axis in axes:
        _finish_axis(axis)
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def plot_binary_repeated(summary, dataset_name, figure, save_path=None):
    """Plot binary Figure 6 or 7 using ordinary curve markers."""
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

    method_styles = {
        "Our Eps-Fair": ("o", "tab:red"),
        "Fairlearn": ("s", "tab:blue"),
        "Fair-adversarial": ("D", "tab:blue"),
    }
    figure_object, axes = plt.subplots(
        1, len(models), figsize=(6 * len(models), 5), squeeze=False
    )
    for axis, model in zip(axes.ravel(), models):
        subset = data[data["Model"] == model]
        for method in methods:
            values = subset[subset["Method"] == method].sort_values(
                "UnfairnessMean"
            )
            marker, color = method_styles[method]
            axis.plot(
                values["UnfairnessMean"],
                values["AccuracyMean"],
                marker=marker,
                color=color,
                label=method,
            )
        unfair = subset[subset["Method"] == "Unfair"]
        axis.scatter(
            unfair["UnfairnessMean"],
            unfair["AccuracyMean"],
            marker="^",
            s=75,
            color="tab:orange",
            label="Unfair",
            zorder=3,
        )
        axis.set(
            xlabel="Unfairness",
            ylabel="Accuracy",
            title=f"{dataset_name} - {model}",
        )
        _finish_axis(axis)
    figure_object.tight_layout()
    _save(figure_object, save_path)
    return figure_object


def plot_multiclass_figure8(summary, save_path=None):
    """Plot multiclass Figure 8 without x/y standard-deviation bars."""
    datasets = ("DRUG", "CRIME")
    models = ("reglog", "RF", "GBM")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, dataset_name in enumerate(datasets):
        for column, model_name in enumerate(models):
            axis = axes[row, column]
            subset = summary[
                (summary["Dataset"] == dataset_name)
                & (summary["Model"] == model_name)
            ]
            curves = (
                (
                    "Our Eps-Fair",
                    "Epsilon",
                    "o",
                    "tab:red",
                    r"$\epsilon$-fair",
                ),
                (
                    "FairProjection",
                    "Alpha",
                    "s",
                    "tab:blue",
                    "FairProjection",
                ),
                (
                    "Fair-transport",
                    "Alpha",
                    "D",
                    "tab:green",
                    "Fair-transport",
                ),
            )
            for method, parameter, marker, color, label in curves:
                values = subset[subset["Method"] == method]
                if method != "Our Eps-Fair" and "OptimizerSuccessRate" in values:
                    values = values[values["OptimizerSuccessRate"] > 0]
                values = values.dropna(
                    subset=["UnfairnessMean", "AccuracyMean"]
                ).sort_values(parameter)
                if values.empty:
                    continue
                axis.plot(
                    values["UnfairnessMean"],
                    values["AccuracyMean"],
                    marker=marker,
                    color=color,
                    label=label,
                )

            unfair = subset[subset["Method"] == "Unfair"].dropna(
                subset=["UnfairnessMean", "AccuracyMean"]
            )
            axis.scatter(
                unfair["UnfairnessMean"],
                unfair["AccuracyMean"],
                marker="^",
                s=75,
                color="tab:orange",
                label="Unfair",
                zorder=3,
            )
            axis.set(
                title=(
                    f"{dataset_name} "
                    f"(K={4 if dataset_name == 'DRUG' else 5}) - {model_name}"
                ),
                xlabel="Unfairness",
                ylabel="Accuracy",
            )
            _finish_axis(axis)

    figure.suptitle(
        "Multi-class real-data experiment: Unfair, epsilon-fair, "
        "FairProjection, Fair-transport"
    )
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def generate_all_figures(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_FIGURE_DIR):
    """Read existing summaries and generate every requested figure except 4."""
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    figure1_summary = _read_summary(
        data_dir / "synthetic" / "figure1_summary.csv"
    )
    figures2_3_summary = _read_summary(
        data_dir / "synthetic" / "figures2_3_summary.csv"
    )
    figure3_distribution_summary = _read_summary(
        data_dir / "synthetic" / "figure3_distribution_summary.csv"
    )
    figure10_summary = _read_summary(
        data_dir / "synthetic" / "figure10_summary.csv"
    )
    drug_binary_summary = _read_summary(
        data_dir / "binary" / "drug_figures6_7_summary.csv"
    )
    crime_binary_summary = _read_summary(
        data_dir / "binary" / "crime_figures6_7_summary.csv"
    )
    multiclass_summary = _read_summary(
        data_dir / "multiclass" / "multiclass_summary_results.csv"
    )

    _paper_style()
    generated = []

    path = output_dir / "figure1.png"
    plot_synthetic_figure1(figure1_summary, path)
    generated.append(path)

    figure2_path = output_dir / "figure2.png"
    figure3_path = output_dir / "figure3.png"
    plot_synthetic_figures2_3(
        figures2_3_summary,
        figure3_distribution_summary,
        figure2_path,
        figure3_path,
    )
    generated.extend((figure2_path, figure3_path))

    path = output_dir / "figure10.png"
    plot_synthetic_figure10(figure10_summary, path)
    generated.append(path)

    for dataset_name, summary in (
        ("DRUG", drug_binary_summary),
        ("CRIME", crime_binary_summary),
    ):
        for figure_number in (6, 7):
            path = output_dir / (
                f"{dataset_name.lower()}_figure{figure_number}.png"
            )
            plot_binary_repeated(
                summary,
                dataset_name,
                figure_number,
                path,
            )
            generated.append(path)

    path = output_dir / "figure8_multiclass.png"
    plot_multiclass_figure8(multiclass_summary, path)
    generated.append(path)

    return generated


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Redraw Figures 1-3, 6-8 and 10 from existing output CSVs "
            "without error-bar crosses. Figure 4 is excluded."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the existing experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Separate destination for the newly rendered PNG files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    generated = generate_all_figures(args.data_dir, args.output_dir)
    print("Generated figures (existing outputs were only read):")
    for path in generated:
        print(f"  {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
