"""
run_visualisation.py
====================

Description
-----
CLI entry point for creating analysis plots from the hyperparameter
search / cross-validation results CSV.

Structure
---------------
- Command-line interface with the following subcommands:
    - hyperparam-search      : overview of hyperparameter search
    - generalization         : train vs test metric scatter (over/underfitting)
    - stability              : stability across folds for top-k configs
    - computational-cost     : test metric vs computation time
    - all                    : reproduce the full default suite of plots

- Core plotting functions:
    - plot_hyperparameter_search_overview
    - generalization_and_overfitting_analysis
    - stability_across_folds_analysis
    - computational_cost_analysis

- Helper to load and pre-process a results CSV:
    - _load_results

How to use with examples
------------------------

1) Simple hyperparameter-search plot

    python run.py visualize-experiments \
        --results-csv outputs/experiments/results_advanced.csv \
        hyperparam-search \
        --metric mean_test_accuracy \
        --hyperparameter lr \
        --hue num_layers \
        --style dense_units \
        --target-metric Accuracy

2) Generalization plot for F1 Macro

    python run.py visualize-experiments \
        --results-csv outputs/experiments/results_advanced.csv \
        generalization \
        --train-metric mean_train_f1_macro \
        --test-metric mean_test_f1_macro \
        --hyperparameter num_layers \
        --target-metric F1_Macro

3) Stability across folds

    python run.py visualize-experiments \
        --results-csv outputs/experiments/results_advanced.csv \
        stability \
        --metric-test mean_test_accuracy \
        --labels-col config_id \
        --top-k 5

4) Computational cost

    python run.py visualize-experiments \
        --results-csv outputs/experiments/results_advanced.csv \
        computational-cost \
        --time-metric mean_fit_time \
        --test-metric mean_test_accuracy \
        --main-hp num_layers \
        --secondary-hp dropout

5) Full default suite of plots

    python run.py visualize-experiments \
        --results-csv outputs/experiments/results_advanced.csv \
        all
"""

###############
# Standard library imports
###############
import argparse
from pathlib import Path
import sys
import random

###############
# Third-party libraries
###############
import pandas as pd
import seaborn as sns

###############
# Global paths and plotting defaults
###############
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "outputs" / "experiments"
sns.set_theme(style="whitegrid")

###############
# Project libraries
###############
from scripts.visualization.relationships import (
    plot_hyperparameter_search_overview,
    generalization_and_overfitting_analysis,
    stability_across_folds_analysis,
    computational_cost_analysis,
)
from utils import set_global_seed

###############
# Helpers: load CSV and prepare columns
###############
def _load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)

    rename_map = {
        "param_clf__batch_size": "batch_size",
        "param_clf__epochs": "epochs",
        "param_clf__model__dense_units": "dense_units",
        "param_clf__model__dropout": "dropout",
        "param_clf__model__learning_rate": "lr",
        "param_clf__model__num_layers": "num_layers",
        "param_clf__model__optimizer": "optimizer",
        "param_clf__model__ridge_penalty" : "lambda"
    }

    present = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=present)

    needed = ["batch_size", "num_layers", "dense_units", "dropout", "lr", "optimizer"]
    if all(col in df.columns for col in needed) and "config_id" not in df.columns:
        df["config_id"] = (
            "bs"
            + df["batch_size"].astype(str)
            + "-L"
            + df["num_layers"].astype(str)
            + "-h"
            + df["dense_units"].astype(str)
            + "-do"
            + df["dropout"].astype(str)
            + "-lr"
            + df["lr"].astype(str)
            + "-opt"
            + df["optimizer"].astype(str)
        )

    return df


###############
# CLI parsing
###############
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from hyperparameter search / CV results."
    )

    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_DIR / "results.csv",
        help="Path to the CSV file containing grid-search / CV results.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=RESULTS_DIR / "plots",
        help="Base directory where plots will be saved.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # 1) Hyperparameter search overview
    hp = sub.add_parser("hyperparam-search", help="Plot hyperparameter search overview.")
    hp.add_argument("--metric", required=True, help="Metric column name (e.g. mean_test_accuracy).")
    hp.add_argument("--hyperparameter", required=True, help="Hyperparameter column to plot on x-axis.")
    hp.add_argument("--hue", required=True, help="Column to use as hue.")
    hp.add_argument("--style", required=True, help="Column to use as style/marker.")
    hp.add_argument(
        "--target-metric",
        default=None,
        help="Name of the metric for titles / filenames (default: use --metric).",
    )
    hp.add_argument("--title", default=None, help="Custom plot title (optional).")
    hp.add_argument("--xlabel", default=None, help="Custom x-axis label (optional).")
    hp.add_argument("--ylabel", default=None, help="Custom y-axis label (optional).")

    # 2) Generalization / overfitting
    gen = sub.add_parser("generalization", help="Generalization vs overfitting analysis.")
    gen.add_argument("--train-metric", required=True, help="Training metric column.")
    gen.add_argument("--test-metric", required=True, help="Test metric column.")
    gen.add_argument("--hyperparameter", required=True, help="Hyperparameter used as color.")
    gen.add_argument(
        "--target-metric",
        default=None,
        help="Name of the metric for titles / filenames.",
    )
    gen.add_argument("--title", default=None, help="Custom plot title (optional).")

    # 3) Stability across folds
    stab = sub.add_parser("stability", help="Stability across CV folds.")
    stab.add_argument("--metric-test", required=True, help="Mean test metric column.")
    stab.add_argument(
        "--labels-col",
        default="config_id",
        help="Column used to label configurations (default: config_id).",
    )
    stab.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top configurations to include (default: 4).",
    )
    stab.add_argument(
        "--target-metric",
        default=None,
        help="Name of the metric for filenames.",
    )

    # 4) Computational cost
    cost = sub.add_parser("computational-cost", help="Accuracy vs computational cost.")
    cost.add_argument("--time-metric", required=True, help="Column with runtime in seconds.")
    cost.add_argument("--test-metric", required=True, help="Test metric column.")
    cost.add_argument("--main-hp", required=True, help="Hyperparameter encoded as color.")
    cost.add_argument("--secondary-hp", required=True, help="Hyperparameter encoded as marker size.")
    cost.add_argument(
        "--target-metric",
        default=None,
        help="Name of the metric for filenames.",
    )

    # 5) All default plots
    all_p = sub.add_parser("all", help="Reproduce the default full suite of plots.")

    return parser.parse_args()


###############
# Dispatcher
###############
def main() -> None:
    set_global_seed()
    args = _parse_args()

    df = _load_results(args.results_csv)
    base_save_dir: Path = args.save_dir

    if args.command == "hyperparam-search":
        target = args.target_metric or args.metric
        title = args.title or f"Effect of {args.hyperparameter} on {target}"
        xlabel = args.xlabel or args.hyperparameter
        ylabel = args.ylabel or target

        save_dir = base_save_dir / "hyperparameter_search"

        plot_hyperparameter_search_overview(
            df=df,
            metric=args.metric,
            hyperparameter=args.hyperparameter,
            hue=args.hue,
            style=args.style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            savepath=save_dir,
            target_metric=target,
        )

    elif args.command == "generalization":
        target = args.target_metric or args.test_metric
        title = args.title or f"Generalization Analysis: {args.hyperparameter}"
        save_dir = base_save_dir / "generalization_and_overfitting"

        generalization_and_overfitting_analysis(
            df=df,
            train_metric=args.train_metric,
            test_metric=args.test_metric,
            hyperparameter=args.hyperparameter,
            title=title,
            savepath=save_dir,
            target_metric=target,
        )

    elif args.command == "stability":
        target = args.target_metric or args.metric_test
        save_dir = base_save_dir / "stability_across_folds"

        stability_across_folds_analysis(
            df=df,
            metric_test=args.metric_test,
            labels_cols=args.labels_col,
            savepath=save_dir,
            top_k=args.top_k,
            target_metric=target,
        )

    elif args.command == "computational-cost":
        target = args.target_metric or args.test_metric
        save_dir = base_save_dir / "computational_cost_analysis"

        computational_cost_analysis(
            df=df,
            time_metric=args.time_metric,
            test_metric=args.test_metric,
            main_hp=args.main_hp,
            secondary_hp=args.secondary_hp,
            savepath=save_dir,
            target_metric=target,
        )

    elif args.command == "all":
        saveplots = base_save_dir
        saveplots_hp_search = saveplots / "hyperparameter_search"
        saveplots_gen_overfit = saveplots / "generalization_and_overfitting"
        saveplots_stability = saveplots / "stability_across_folds"
        saveplots_comp_cost = saveplots / "computational_cost_analysis"

        train_test_metric_accuracy = ("mean_train_accuracy", "mean_test_accuracy", "acc")
        train_test_metric_f1 = ("mean_train_f1_macro", "mean_test_f1_macro", "F1_Macro")

        for general_metric_train, general_metric_test, target_metric in [
            train_test_metric_accuracy,
            train_test_metric_f1,
        ]:
            # Learning rate
            plot_hyperparameter_search_overview(
                df=df,
                metric=general_metric_test,
                hyperparameter="lr",
                hue="num_layers",
                style="dense_units",
                title=f"Effect of Learning Rate on Test {target_metric}",
                xlabel="Learning Rate",
                ylabel=f"{target_metric}",
                savepath=saveplots_hp_search,
                target_metric=target_metric,
            )

            # Dropout
            plot_hyperparameter_search_overview(
                df=df,
                metric=general_metric_test,
                hyperparameter="dropout",
                hue="num_layers",
                style="dense_units",
                title=f"Effect of Dropout on Test {target_metric}",
                xlabel="Dropout Rate",
                ylabel=f"{target_metric}",
                savepath=saveplots_hp_search,
                target_metric=target_metric,
            )

            # Dense units
            plot_hyperparameter_search_overview(
                df=df,
                metric=general_metric_test,
                hyperparameter="dense_units",
                hue="num_layers",
                style="batch_size",
                title=f"Effect of Dense Units on Test {target_metric}",
                xlabel="Dense Units",
                ylabel=f"{target_metric}",
                savepath=saveplots_hp_search,
                target_metric=target_metric,
            )

            # Generalization / overfitting
            generalization_and_overfitting_analysis(
                df=df,
                train_metric=general_metric_train,
                test_metric=general_metric_test,
                hyperparameter="num_layers",
                title="Generalization Analysis : Number of Layers",
                savepath=saveplots_gen_overfit,
                target_metric=target_metric,
            )

            # Stability
            stability_across_folds_analysis(
                df=df,
                metric_test=general_metric_test,
                labels_cols="config_id",
                savepath=saveplots_stability,
                top_k=4,
                target_metric=target_metric,
            )

            # Computational cost
            time_metric = "mean_fit_time"
            computational_cost_analysis(
                df=df,
                time_metric=time_metric,
                test_metric=general_metric_test,
                main_hp="num_layers",
                secondary_hp="dropout",
                savepath=saveplots_comp_cost,
                target_metric=target_metric,
            )

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()