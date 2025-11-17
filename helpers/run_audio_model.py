"""
run_audio_model.py
=======================

High-level CLI entry point to **train** and **evaluate** a speech–emotion
classifier from the project root.

This script coordinates two steps for a given model family:

    1) Training:
           - calls `train_model(model_path, infile)` from a model-specific
             training module (e.g. `audio_model_baseline.py` for the baseline).
           - saves the trained Keras model parameters to `model_path`.

    2) Prediction:
           - calls `audio_pred(model_path, infile, outfile)` from a model-specific
             prediction module (e.g. `audio_prediction_baseline.py` for the baseline).
           - reads the same feature CSV (or another, if you pass it) and writes
             a CSV of predicted emotions.

Model families
--------------

The script is designed to support multiple model variants with the same API:

    train_model(model_path: str, infile: str) -> (model, int2emotions, emotions2int)
    audio_pred(model_path: str, infile: str, outfile: str) -> None

By default (`--model-type baseline`), it imports:

    - `audio_model_baseline`        (must define `train_model`)
    - `audio_prediction_baseline`   (must define `audio_pred`)

For other model types, it expects modules named:

    - `audio_model_<model_type>`
    - `audio_prediction_<model_type>`

Directory and file handling
---------------------------

The script takes an input CSV of features (`--infile`) and an output directory
(`--output-dir`) where both the model parameters and predictions will be stored.

The output files follow this naming convention:

    <output-dir>/
        <model_type>_voice_model.h5
        <model_type>_predictions.csv

where `<model_type>` is the value of `--model-type` (e.g. "baseline", "cnn").

Usage examples
--------------

1) Baseline model, custom input and output directory:

    python helpers/run_audio_model.py \
        --infile /data/merged_with_features.csv \
        --output-dir /data/emotion_runs/baseline

3) Alternative model type (e.g. "cnn") with its own modules:

    python run_emotion_pipeline.py \
        --model-type cnn \
        --infile /data/merged_with_features.csv \
        --output-dir /data/emotion_runs/cnn

Requirements
------------

- The selected training and prediction modules must be importable from ROOT:
      baseline:
          audio_model_baseline.py        (train_model)
          audio_prediction_baseline.py   (audio_pred)

      other model types:
          audio_model_<type>.py
          audio_prediction_<type>.py

- The CSV passed via `--infile` must have the same feature layout as expected
  by the chosen model’s training/prediction code.
"""

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def _load_model_functions(model_type: str):
    """
    Dynamically import `train_model` and `audio_pred` for a given model type.

    Parameters
    ----------
    model_type : str
        Name of the model family. "baseline" uses `audio_model` and
        `audio_prediction`. Any other value expects modules named
        `audio_model_<model_type>` and `audio_prediction_<model_type>`.

    Returns
    -------
    (train_model, audio_pred) : tuple[callable, callable]
        Callable training and prediction functions.
    """
    if model_type == "baseline":
        from scripts.models.audio_model_baseline import train_model  # type: ignore
        from scripts.models.audio_prediction_baseline import audio_pred  # type: ignore
        return train_model, audio_pred

    # For custom models, follow the naming convention:
    #   audio_model_<model_type>.py
    #   audio_prediction_<model_type>.py
    train_module_name = f"audio_model_{model_type}"
    pred_module_name = f"audio_prediction_{model_type}"

    train_module = importlib.import_module(train_module_name)
    pred_module = importlib.import_module(pred_module_name)

    train_model = getattr(train_module, "train_model")
    audio_pred = getattr(pred_module, "audio_pred")

    return train_model, audio_pred


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for training and prediction.

    Returns
    -------
    argparse.Namespace
        Namespace with attributes:
            infile      : Path
            output_dir  : Path
            metrics_dir : Path
            model_type  : str
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a speech–emotion model and generate predictions from a "
            "pre-extracted feature CSV."
        )
    )

    parser.add_argument(
        "--infile",
        type=Path,
        required=True,
        help=(
            "Path to the CSV with input features. "
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help=(
            "Directory where model parameters and prediction CSV will be stored. "
            "Files will be named '<model_type>_voice_model.h5' and "
            "'<model_type>_predictions.csv'. "
            "Default: <repo>/outputs"
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=ROOT / "outputs" / "metrics.csv",
        help=(
            "Directory where model metrics CSV will be stored. "
            "Default: <repo>/outputs"
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        help=(
            "Model family to use. 'baseline' expects 'audio_model.py' and "
            "'audio_prediction.py'. Any other value expects modules named "
            "'audio_model_<model_type>.py' and 'audio_prediction_<model_type>.py'. "
            "Default: baseline"
        ),
    )

    return parser.parse_args()


def main_train(train_model_fn, model_path: Path, infile: Path, metrics_file:Path) -> None:
    """
    Wrapper around the model-specific `train_model` function.

    Parameters
    ----------
    train_model_fn : callable
        Function with signature `train_model(model_path, infile)`.
    model_path : Path
        Destination path for the trained model file (.h5).
    infile : Path
        Input CSV path with training features.
    outfile : Path
        Output folder where the metrics will be stored.
    """
    train_model_fn(str(model_path), str(infile), str(metrics_file))


def main_test(audio_pred_fn, model_path: Path, infile: Path, outfile: Path) -> None:
    """
    Wrapper around the model-specific `audio_pred` function.

    Parameters
    ----------
    audio_pred_fn : callable
        Function with signature `audio_pred(model_path, infile, outfile)`.
    model_path : Path
        Path to the trained model file (.h5).
    infile : Path
        Input CSV path with features for prediction.
    outfile : Path
        Output CSV path where predictions will be written.
    """
    audio_pred_fn(str(model_path), str(infile), str(outfile))


def main() -> None:
    args = _parse_args()

    infile: Path = args.infile
    output_dir: Path = args.output_dir
    model_type: str = args.model_type
    metrics_dir: Path = args.metrics_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model and prediction file paths
    model_path = output_dir / f"{model_type}_voice_model.h5"
    predictions_path = output_dir / f"{model_type}_predictions.csv"

    # Load model-specific training and prediction functions
    train_model_fn, audio_pred_fn = _load_model_functions(model_type)

    # Train and save model
    main_train(train_model_fn, model_path, infile, metrics_dir)

    # Run predictions and save output
    main_test(audio_pred_fn, model_path, infile, predictions_path)


if __name__ == "__main__":
    main()
