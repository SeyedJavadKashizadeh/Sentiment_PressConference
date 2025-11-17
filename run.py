"""
run.py
======

Top-level pipeline entry point for the project.

This script acts as a thin dispatcher to the existing, more specialized CLIs:

    - run_extract_training_features.py
    - run_extract_fomc_features.py
    - run_audio_model.py

Available subcommands
---------------------
1) extract-training-features
   Wraps: `run_extract_training_features.py`

   Purpose:
        Ensure raw training datasets are downloaded. If not, download them.
        Then filters the datasets to keep the desired emotions and gender/s.
        Merge all the datasets into a csv file containing dataset name | Path to audio | Emotion name | Emotion Id
        Extracts features of librosa or/and opensmile
        Generate a new csv file which extends the previous one with the features extracted

   Typical usage:

       python run.py extract-training-features \
           --raw-data-dir   data_training/raw
           --in-file        data_training/merged.csv \
           --out-file       data_training/merged_with_features_librosa.csv \
           --engine         both
           --gender         None --> it takes male and female
           --emotions       1 2 3 4 5 6 7 8 <=> ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "pleasant_surprise"]
           --manifest       data_training/merged.csv
           --conversion_dir data_training/converted_16khz

2) extract-fomc-features
   Wraps: `run_extract_fomc_features.py`

   Purpose:
       Download the FedSentimentLab FOMC audio dataset from Hugging Face,
       convert to mono/16 kHz, and extract acoustic features (Librosa,
       OpenSMILE, or both), writing Parquet feature tables.

   Typical usage:

       python run.py extract-fomc-features \
           --download-dir data_fomc/raw_data \
           --convert-dir  data_fomc/converted_16khz \
           --features-dir data_fomc/features \
           --engine       librosa

3) train-model
   Wraps: `run_audio_model.py`

   Purpose:
       Train a speech–emotion classifier on a feature CSV (e.g. the training
       features produced above) and write both model weights and predictions
       on the training set. Supports multiple model “families” via `--model-type`.

   Typical usage:

       python run.py train-model \
           --infile     data_training/merged_with_features_librosa.csv \
           --output-dir outputs \
           --model-type baseline

How to use
----------

General pattern:

    python run.py <subcommand> [options for that subcommand...]

Implementation notes
--------------------

- `run.py` uses `argparse` with a single positional `command` and forwards the
  remaining arguments (`argparse.REMAINDER`) to the underlying script.
- It invokes each script using the same Python interpreter (`sys.executable`).
- Paths to scripts are resolved relative to the location of `run.py` so you
  can run it from anywhere as long as the working directory is inside the repo.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

HELPERS = "helpers"

# Map subcommand names to script filenames (relative to repo root)
SUBCOMMANDS = {
    "assemble-training": f"{HELPERS}/assemble_training_data.py",
    "extract-training-features": f"{HELPERS}/run_extract_training_features.py",
    "extract-fomc-features": f"{HELPERS}/run_extract_fomc_features.py",
    "train-model": f"{HELPERS}/run_audio_model.py",
}

def _parse_args() -> argparse.Namespace:
    """
    Parse the top-level arguments for run.py.

    Returns
    -------
    argparse.Namespace
        Namespace with:
            command : str
                One of SUBCOMMANDS.keys()
            args    : List[str]
                Remaining arguments to forward to the underlying script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Project pipeline entry point. Dispatches to specialized scripts "
            "such as assemble_training_data.py, run_extract_training_features.py, "
            "run_extract_fomc_features.py, and run_audio_model.py."
        )
    )

    parser.add_argument(
        "command",
        choices=sorted(SUBCOMMANDS.keys()),
        help=(
            "Pipeline step to run. Use one of: "
            + ", ".join(sorted(SUBCOMMANDS.keys()))
        ),
    )

    # All remaining args are passed verbatim to the underlying script
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to the selected subcommand script.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    script_name = SUBCOMMANDS[args.command]

    # Resolve script path relative to this file’s directory
    script_path = ROOT / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script for command '{args.command}' not found at: {script_path}"
        )

    # Build command: python <script> <forwarded args...>
    cmd = [sys.executable, str(script_path)] + args.args

    # Run the underlying script as a subprocess
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
