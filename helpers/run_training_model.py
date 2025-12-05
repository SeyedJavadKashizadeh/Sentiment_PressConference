"""
run_training_model.py
=====================

Description
-------------------------
Train a baseline or advanced neural network for speech–emotion recognition
on a feature CSV, using the shared model factory and data splitting utilities.

Structure
-------------------------------
- CLI interface with arguments:
    --infile              : path to CSV with training features
    --model-path          : where to save the trained model (.keras)
    --metrics-csv         : optional CSV to log global metrics (e.g. accuracy)
    --mode                : "baseline" (paper) or "advanced" (extended)
    --num-layers          : override number of dense blocks (advanced)
    --dense-units         : override hidden units per layer
    --dropout             : override dropout rate
    --optimizer           : optimizer name (adam, rmsprop, sgd)
    --learning-rate       : learning rate
    --use-batchnorm       : add batch normalization layers
    --activation          : activation name ("relu", "gelu", etc.)
    --ridge-penalty       : ridge penalty to add between layers
    --lasso-penalty       : lasso penalty to add between layers
    --standardize-inputs  : if passed, force standardization of inputs

- Core steps:
    1. Parse CLI arguments and build a configuration dict.
    2. Load and split data with `split_data`.
    3. Optionally standardize features and save a scaler.
    4. Build a model via `create_model`.
    5. Train with callbacks (checkpoint + LR scheduler).
    6. Evaluate on the held-out test set and save metrics/model.

How to use with examples
------------------------

Train baseline (paper-like) model:

    python run.py train-model \
        --mode baseline \
        --infile data_training/merged_with_features.csv \
        --model-path outputs/baseline/baseline_model.keras \
        --metrics-csv outputs/baseline/baseline_metrics.csv

Train advanced model with custom hyperparameters:

    python run_training_model.py \
    --mode advanced \
    --infile data_training/merged_with_features.csv \
    --model-path outputs/both/advanced_tess_emodb.keras \
    --metrics-csv outputs/both/advanced_tess_emodb_metrics.csv \
    --datasets TESS EmoDB \
    --num-layers 2 \
    --dense-units 512 \
    --dropout 0.1 \
    --optimizer rmsprop \
    --learning-rate 0.001 \
    --batch-size 128 \
    --epochs 600 \
    --ridge-penalty 1e-5 \
    --lasso-penalty 1e-5 \
    --activation gelu \
    --use-batchnorm \
    --standardize-inputs
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import joblib

###############
# Project imports
###############
from scripts.models.model_factory import (
    EMOTIONS_BASELINE,
    EMOTIONS_ADVANCED,
    BASELINE_CONFIG,
    ADVANCED_DEFAULT_CONFIG,
    create_model,
)
from scripts.models.utils_models import split_data
from scripts.visualization.training_visualization import (
    plot_confusion_matrix_heatmap,
    plot_history,
    plot_per_class_metrics_bars,
    plot_per_dataset_f1_bar,
    compute_per_dataset_metrics
    )

from utils import set_global_seed
###############
# Argument parsing
###############
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline or advanced model on a given dataset."
    )
    parser.add_argument(
        "--infile",
        type=Path,
        required=True,
        help="CSV with training features.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Where to save the trained weights (.keras).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV file to store global metrics.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced"],
        default="baseline",
        help="Baseline (paper) or advanced (extended) configuration.",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dense-units", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-batchnorm", action="store_true")
    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--ridge-penalty", type=float, default=0.0)
    parser.add_argument("--lasso-penalty", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--standardize-inputs",
        action="store_true",
        help="If set, force standardization of inputs (overrides config).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional list of dataset names to keep (values in 'dataset' column), "
            "e.g. --datasets TESS EmoDB RAVDESS."
        ),
    )

    return parser.parse_args()


###############
# Build configuration from mode + CLI overrides
###############
def _build_config(mode: str, args: argparse.Namespace) -> dict:
    if mode == "baseline":
        cfg = BASELINE_CONFIG.copy()
    else:
        # advanced starts from baseline + advanced defaults
        cfg = {**BASELINE_CONFIG, **ADVANCED_DEFAULT_CONFIG}

    # Apply overrides from CLI (only if provided)
    if args.num_layers is not None:
        cfg["num_layers"] = args.num_layers
    if args.dense_units is not None:
        cfg["dense_units"] = args.dense_units
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.optimizer is not None:
        cfg["optimizer"] = args.optimizer
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.use_batchnorm:
        cfg["use_batchnorm"] = True
    if args.activation is not None:
        cfg["activation"] = args.activation
    if args.standardize_inputs:
        cfg["standardize"] = True
    if args.ridge_penalty is not None:
        cfg["ridge_penalty"] = args.ridge_penalty
    if args.lasso_penalty is not None:
        cfg["lasso_penalty"] = args.lasso_penalty
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    return cfg

###############
# Main training routine
###############
def main() -> None:
    set_global_seed()
    args = _parse_args()

    ### Filter datasets
    # If specific datasets are requested, filter the input CSV on 'dataset' column
    effective_infile = args.infile
    if args.datasets is not None:
        df = pd.read_csv(args.infile)

        # Assuming the column is exactly called 'dataset'
        # and values match the strings passed in --datasets (e.g. "TESS", "EmoDB", "RAVDESS")
        mask = df["dataset"].isin(args.datasets)
        df_filtered = df[mask].copy()

        if df_filtered.empty:
            raise ValueError(
                f"No rows left after filtering for datasets={args.datasets}. "
                "Check the 'dataset' column values in the CSV."
            )

        # Write filtered data to a temporary CSV next to the original file
        filtered_suffix = "_".join(args.datasets)
        effective_infile = args.infile.with_name(
            f"{args.infile.stem}_filtered_{filtered_suffix}{args.infile.suffix}"
        )
        df_filtered.to_csv(effective_infile, index=False)
        print(f"[INFO] Filtered training data written to: {effective_infile}")

    ### Load and split data
    ## Balanced split using split_data
    # Specification: uses EMOTIONS defined in model_factory.
    # Choose label set depending on mode
    if args.mode == "baseline":
        emotions = EMOTIONS_BASELINE
    else:  # "advanced"
        emotions = EMOTIONS_ADVANCED

    int2emotions, emotions2int, x_train, y_train, x_test, y_test, ds_train, ds_test = split_data(
        emotions=emotions,
        filename=effective_infile,
        train_set=0.8
    )
    
    y_train = np.array([emotions2int[label] for label in y_train.ravel()])
    y_test  = np.array([emotions2int[label] for label in y_test.ravel()])

    target_class = len(emotions)
    input_length = x_train.shape[1]

    ### Build configuration
    ## Merge mode defaults with CLI overrides
    cfg = _build_config(args.mode, args)
    print(f"Training mode={args.mode} with config:\n{cfg}")

    ### Optional standardization
    ## Fit StandardScaler on train and apply to train/test
    # Specification: scaler is saved alongside the model with .scaler.pkl.
    if cfg.get("standardize", False):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        scaler_path = args.model_path.with_suffix(".scaler.pkl")
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[saved scaler] {scaler_path}")

    ### Build model from factory
    ## Only pass architecture/optimizer keys to create_model
    model = create_model(
        input_length=input_length,
        target_class=target_class,
        num_layers=cfg["num_layers"],
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
        optimizer=cfg["optimizer"],
        learning_rate=cfg["learning_rate"],
        use_batchnorm=cfg["use_batchnorm"],
        activation=cfg["activation"],
        ridge_penalty=cfg["ridge_penalty"],
        lasso_penalty=cfg["lasso_penalty"]
    )

    ### Train model
    ## Callbacks: checkpoint + ReduceLROnPlateau
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = ModelCheckpoint(
        str(args.model_path),
        save_best_only=True,
        monitor="val_loss",
    )

    lr_reduce = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.9,
        patience=20,
        min_lr=1e-6,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=cfg.get("batch_size", 64),
        epochs=cfg.get("epochs", 1000),
        validation_data=(x_test, y_test),
        callbacks=[checkpointer, lr_reduce],
        verbose=1,
    )

    df_hist = pd.DataFrame(history.history)
    loss_png_path = args.model_path.with_suffix(".loss.png")
    acc_png_path  = args.model_path.with_suffix(".accuracy.png")
    f1_png_path   = args.model_path.with_suffix(".f1.png")
    plot_history(df_hist, loss_png_path, acc_png_path, f1_png_path)

    ### Evaluation on test set
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    y_true = y_test
    acc = accuracy_score(y_true, y_pred)

    ## Confusion matrix (counts)
    label_indices = [emotions2int[e] for e in emotions]
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (counts):")
    print(cm)

    ## Macro and per-class metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=label_indices
    )

    print(
        f"Macro scores - Precision: {prec_macro:.4f}, "
        f"Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}"
    )

    ## Plots
    cm_png    = args.model_path.with_suffix(".cm.png")
    metrics_png = args.model_path.with_suffix(".per_class_metrics.png")

    plot_confusion_matrix_heatmap(
        cm=cm,
        emotions=emotions,
        outfile=cm_png,
        normalize=True,
    )
    plot_per_class_metrics_bars(
        emotions=emotions,
        precision=prec_pc,
        recall=rec_pc,
        f1=f1_pc,
        outfile=metrics_png,
    )

    ### Save metrics
    if args.metrics_csv is not None:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_global = pd.DataFrame(
            {
                "accuracy": [acc],
                "precision_macro": [prec_macro],
                "recall_macro": [rec_macro],
                "f1_macro": [f1_macro],
            }
        )
        metrics_global.to_csv(args.metrics_csv, index=False)
        print(f"[saved global metrics] {args.metrics_csv}")

        per_class_csv = args.metrics_csv.with_suffix(".per_class.csv")
        metrics_per_class = pd.DataFrame(
            {
                "emotion": emotions,
                "precision": prec_pc,
                "recall": rec_pc,
                "f1": f1_pc,
            }
        )
        metrics_per_class.to_csv(per_class_csv, index=False)
        print(f"[saved per-class metrics] {per_class_csv}")

    ### Per dataset
    df_ds_metrics = compute_per_dataset_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ds_test=ds_test,
        emotions=emotions,
        emotions2int=emotions2int,
    )

    if args.metrics_csv is not None:
        per_ds_csv = args.metrics_csv.with_suffix(".per_dataset.csv")
    else:
        per_ds_csv = args.model_path.with_suffix(".per_dataset.csv")

    df_ds_metrics.to_csv(per_ds_csv, index=False)
    print(f"[saved per-dataset metrics] {per_ds_csv}")

    per_ds_png = per_ds_csv.with_suffix(".png")
    plot_per_dataset_f1_bar(df_ds_metrics, per_ds_png)

if __name__ == "__main__":
    main()