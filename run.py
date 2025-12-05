"""
run.py
======

Top-level pipeline entry point for the project.

Description
-----
This script is a thin dispatcher that forwards subcommands to specialized
helper CLIs living under the `helpers/` folder. It centralizes the way you
interact with the full pipeline (data preparation, feature extraction,
model training, prediction, cross-validation, and experiment visualization).

Structure
---------------
- A mapping from **subcommand names** to Python helper scripts:
    - helpers/run_extract_training_features.py
    - helpers/run_extract_fomc_features.py
    - helpers/run_training_model.py
    - helpers/run_predict_model.py
    - helpers/run_cross_validation.py
    - helpers/run_visualisation.py

- A simple `argparse` interface:
    - `command`: which pipeline step to run
    - `args`: remaining arguments forwarded verbatim to the chosen helper

- A small wrapper that:
    - resolves the correct script path relative to this file
    - invokes it with the same Python interpreter (`sys.executable`)

Available subcommands and examples
----------------------------------
0) eda-analysis
    Wraps: 'helpers/run_eda_analysis.py'

    Purpose:
        Operate an EDA on the training datasets RAVDESS / EmoDB / TESS.
        The script is run-only, no arguments are specified other than
        the source of the features df for librosa and opensmile.
    
    Example:
        python run.py eda-analysis \
            --librosa-csv  data_training/merged_with_features_librosa.csv \
            --opensmile-csv data_training/merged_with_features_opensmile.csv \
            --save-dir     outputs/eda_analysis

1) extract-training-features
   Wraps: `helpers/run_extract_training_features.py`

   Purpose:
       Ensure raw training datasets are present (download if needed),
       assemble a manifest CSV across datasets, and extend it with
       Librosa/OpenSMILE features.

   Example:

       python run.py extract-training-features \
           --raw-data-dir   data_training/raw_data \
           --manifest       data_training/merged.csv \
           --out-file       data_training/merged_with_features.csv \
           --engine         both \
           --gender         female \
           --emotions       happy fear

2) extract-fomc-features
   Wraps: `helpers/run_extract_fomc_features.py`

   Purpose:
       Download the FedSentimentLab FOMC audio dataset from Hugging Face,
       convert to mono/16 kHz, and extract acoustic features (Librosa,
       OpenSMILE, or both) into Parquet files.

   Example:

       python run.py extract-fomc-features \
           --download-dir data_fomc/raw_data \
           --convert-dir  data_fomc/converted_16khz \
           --features-dir data_fomc/features \
           --engine       librosa

3) train-model
   Wraps: `helpers/run_training_model.py`

   Purpose:
       Train a **baseline** or **advanced** emotion model on a feature CSV.
       Handles train/test split, optional standardization, and saves model
       weights and optional metrics.

   Examples:

       # Baseline (paper-style) model
       python run.py train-model \
           --infile      data_training/merged_with_features.csv \
           --model-path  outputs/models/baseline.h5 \
           --metrics-csv outputs/metrics/baseline_metrics.csv \
           --mode        baseline

       # Advanced model with custom hyperparameters
       python run.py train-model \
           --infile      data_training/merged_with_features.csv \
           --model-path  outputs/models/advanced.h5 \
           --metrics-csv outputs/metrics/advanced_metrics.csv \
           --mode        advanced \
           --num-layers  3 \
           --dense-units 256 \
           --dropout     0.3 \
           --optimizer   adam \
           --learning-rate 1e-3 \
           --use-batchnorm \
           --standardize-inputs True

4) predict-model
   Wraps: `helpers/run_predict_model.py`

   Purpose:
       Load a trained `.h5` model (and optional `.scaler.pkl`), run
       predictions on a feature CSV, and save the predicted emotions.

   Example:

       python run.py predict-model \
           --weights  outputs/models/baseline.h5 \
           --infile   data_fomc/features_merged.parquet \
           --outfile  outputs/predictions/baseline_fomc_predictions.csv

5) cross-validation
   Wraps: `helpers/run_cross_validation.py`

   Purpose:
       Run cross-validation + grid search for the **advanced** model,
       using the shared model factory and standardized inputs. Saves
       CV results and the best model.

   Example:

       python run.py cross-validation \
           --training-dataset data_training/merged_with_features.csv \
           --cv-splits        3 \
           --seed             42

6) visualize-experiments
   Wraps: `helpers/run_visualisation.py`

   Purpose:
       Generate hyperparameter search and diagnostic plots from a
       `results*.csv` file (e.g. from cross-validation).

   Example:

       python run.py visualize-experiments hyperparam-search \
           --results-csv   outputs/experiments/results_advanced.csv \
           --metric        mean_test_accuracy \
           --hyperparameter lr \
           --hue           num_layers \
           --style         dense_units \
           --target-metric Accuracy

How to use
----------
General pattern:

    python run.py <subcommand> [options for that subcommand...]

For example, to train a baseline model and then predict FOMC emotions:

    python run.py train-model \
        --infile     data_training/merged_with_features.csv \
        --model-path outputs/models/baseline.h5 \
        --mode       baseline

    python run.py predict-model \
        --weights outputs/models/baseline.h5 \
        --infile  data_fomc/features_merged.parquet \
        --outfile outputs/predictions/baseline_fomc_predictions.csv
"""

###############
# Standard library imports
###############
import argparse
import subprocess
import sys
from pathlib import Path
from utils import set_global_seed
###############
# Paths and subcommand mapping
###############
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

HELPERS = "helpers"

# Map subcommand names to script filenames (relative to repo root)
SUBCOMMANDS = {
    "extract-training-features": f"{HELPERS}/run_extract_training_features.py",
    "extract-fomc-features": f"{HELPERS}/run_extract_fomc_features.py",
    "train-model": f"{HELPERS}/run_training_model.py",
    "predict-model": f"{HELPERS}/run_predict_model.py",
    "cross-validation": f"{HELPERS}/run_cross_validation.py",
    "visualize-experiments": f"{HELPERS}/run_visualisation.py",
    "eda-analysis": f"{HELPERS}/run_eda_analysis.py",
}

###############
# Argument parsing
###############
def _parse_args() -> argparse.Namespace:
    """
    Parse the top-level arguments for run.py.

    Returns
    -------
    argparse.Namespace
        Fields:
            - command : str
                One of the keys in SUBCOMMANDS.
            - args    : List[str]
                Remaining arguments to forward to the underlying helper script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Top-level pipeline entry point. Dispatches to helper scripts under "
            "the 'helpers/' directory (feature extraction, training, prediction, "
            "cross-validation, and visualisation)."
        )
    )

    parser.add_argument(
        "command",
        choices=sorted(SUBCOMMANDS.keys()),
        help=(
            "Pipeline step to run. Available: "
            + ", ".join(sorted(SUBCOMMANDS.keys()))
        ),
    )

    # All remaining args are passed verbatim to the underlying script
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to the selected helper script.",
    )

    return parser.parse_args()


###############
# Main dispatcher
###############
def main() -> None:
    set_global_seed()

    args = _parse_args()

    script_name = SUBCOMMANDS[args.command]

    ### Resolve script path
    ## From relative name to absolute path
    # Specification: ensure we locate the helper script relative to this file.
    script_path = ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script for command '{args.command}' not found at: {script_path}"
        )

    ### Build and run subprocess command
    ## Use same Python interpreter
    # Specification: run `python <helper> <args...>` with error propagation.
    cmd = [sys.executable, str(script_path)] + args.args

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
