"""
run_predict_model.py
====================

Description
-------------------------
Predict emotions for a given feature table using a trained baseline or
advanced model. Optionally applies the same StandardScaler that was
fitted at training time.

Structure
-------------------------------
- CLI interface with arguments:
    --mode      : "baseline" or "advanced" emotion set
    --weights   : path to trained Keras model (.keras file)
    --infile    : CSV or Parquet file with features to predict on
    --outfile   : path to CSV file where predictions will be stored

- Core steps:
    1. Load features (CSV or Parquet).
    2. Apply the same column filtering used at training time.
    3. Optionally load and apply a StandardScaler (.scaler.pkl) if present.
    4. Load the trained model (.keras).
    5. Run predictions and map class indices to emotion labels.
    6. Save predictions to a CSV with columns: item, emotion.

How to use with examples
------------------------

Baseline model prediction:

    python run.py predict-model \
        --weights outputs/baseline/baseline_model.keras \
        --infile  data_fomc/features/features_merged.parquet \
        --outfile outputs/baseline/baseline_predictions.csv

Advanced model prediction:

    python run.py predict-model \
        --weights outputs/both/advanced_model.keras \
        --infile  data_fomc/features/features_merged.parquet \
        --outfile outputs/both/advanced_predictions.csv
"""

###############
# Standard libraries
###############
import argparse
from pathlib import Path
import sys

###############
# Add project root to sys.path
###############
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

###############
# Third-party libraries
###############
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

###############
# Project imports
###############
from scripts.models.model_factory import EMOTIONS_ADVANCED, EMOTIONS_BASELINE
from utils import set_global_seed
###############
# Argument parsing
###############
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict emotions for a feature table (CSV or Parquet)."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced"],
        default="baseline",
        required=True,
        help="Which label set to use when mapping class indices to emotions.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Trained Keras model (.keras).",
    )
    parser.add_argument(
        "--infile",
        type=Path,
        required=True,
        help="CSV or Parquet file with features to predict.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        required=True,
        help="Where to save predictions CSV.",
    )
    return parser.parse_args()


###############
# Feature preprocessing
###############
def processing_data(df: pd.DataFrame) -> np.ndarray:
    """
    Apply the same feature selection logic used during training:
    drop identifier and non-feature columns, keep only numeric features.
    """
    df = df.drop([col for col in df.columns if "item" in col], axis=1)
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion" in col], axis=1)

    x = df.to_numpy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


###############
# Main prediction routine
###############
def main() -> None:
    set_global_seed()
    args = _parse_args()

    ### Load input features
    ## Support CSV and Parquet
    if args.infile.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(args.infile, sep=",")

    x_pred = processing_data(df)

    ### Load scaler if present
    ## Keep training-time preprocessing consistent
    scaler_path = args.weights.with_suffix(".scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        x_pred = scaler.transform(x_pred)
        print(f"[loaded scaler] {scaler_path}")
    else:
        print("[info] No scaler file found; using raw features (baseline or non-standardized model).")

    ### Load model and run predictions
    ## Use EMOTIONS from model_factory for label mapping
    if args.mode == "baseline":
        emotions = EMOTIONS_BASELINE
    else:
        emotions = EMOTIONS_ADVANCED

    idx2emo = {i: e for i, e in enumerate(emotions)}

    model = load_model(args.weights)
    y_pred = np.argmax(model.predict(x_pred), axis=-1)

    ### Build and save output CSV
    ## Columns: item (row index), emotion (string label)
    out = pd.DataFrame(
        {
            "item": df.index,
            "emotion": pd.Series(y_pred).map(idx2emo),
        }
    )
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outfile, sep=",", index=False)
    print(f"[saved] {args.outfile}")


if __name__ == "__main__":
    main()
