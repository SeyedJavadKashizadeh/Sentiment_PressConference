"""
run_cross_validation.py
=======================

Description
-------------------------
Run cross-validation and hyperparameter grid search for the **advanced**
neural network model used for speech–emotion recognition.

Structure
-------------------------------
- Loads a feature CSV (typically produced by `run_extract_training_features.py`)
- Splits data into train/test using `split_data` (balanced across emotions)
- Standardizes features inside a sklearn `Pipeline`
- Wraps the Keras model with `SciKeras` for use in `GridSearchCV`
- Tunes a predefined hyperparameter grid for the advanced model
- Saves:
    - `outputs/experiments/results_advanced.csv`
    - `outputs/experiments/advanced_best_model_in_CV.keras`

Command-line arguments:
    --training-dataset   : path to CSV with training features
    --cv-splits          : number of StratifiedKFold splits
    --seed               : random seed for splitting and CV

How to use with examples
------------------------

Basic usage (default file + settings):

    python run.py cross-validation

Custom dataset and fewer splits:

    python run.py cross-validation \
        --training-dataset data_training/merged_with_features.csv \
        --cv-splits 3 \
        --seed 123
"""

###############
# Standard libraries
###############
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Sequence, Mapping, List

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
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

###############
# Project paths
###############
RESULTS_DIR = ROOT / "outputs" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

###############
# Project imports
###############
from scripts.models.utils_models import split_data  # type: ignore
from scripts.models.model_factory import (          # type: ignore
    EMOTIONS_ADVANCED,
    create_model,
    ADVANCED_DEFAULT_CONFIG,
)
from utils import set_global_seed
###############
# Hyperparameter grid for advanced model
###############
ADVANCED_PARAM_GRID: Dict[str, Any] = {
    "clf__model__num_layers":    [6],
    "clf__model__dense_units":   [128,256,512],
    "clf__model__dropout":       [0.1, 0.3],
    "clf__model__optimizer":     ["rmsprop", "adam"],
    "clf__model__learning_rate": [1e-3, 3e-4],
    "clf__batch_size":           [32, 64, 128],
    "clf__epochs":               [80],
    "clf__model__use_batchnorm": [True],
    "clf__model__activation":    ["gelu"],
    "clf__model__ridge_penalty": [1e-8, 1e-5],
    "clf__model__lasso_penalty": [1e-8, 1e-5]
}


PARAM_KEYS = list(ADVANCED_PARAM_GRID.keys())
PARAM_COLS = [f"param_{k}" for k in PARAM_KEYS]

def build_filtered_param_grid(results_csv: Path) -> List[Dict[str, list]]:
    # All combinations as scalar-valued dicts
    all_param_dicts = list(ParameterGrid(ADVANCED_PARAM_GRID))

    if not results_csv.exists():
        # Convert each combination into a valid "small grid" dict
        return [{k: [v] for k, v in p.items()} for p in all_param_dicts]

    existing = pd.read_csv(results_csv)

    # If previous file has no param columns (e.g., different structure), just run all
    if not set(PARAM_COLS).issubset(existing.columns):
        return [{k: [v] for k, v in p.items()} for p in all_param_dicts]

    done = existing[PARAM_COLS].drop_duplicates()

    def is_done(p: Dict[str, Any]) -> bool:
        # build a row matching the param columns order
        values = [p[k] for k in PARAM_KEYS]
        # done.values is (n_rows, n_params), values is (n_params,)
        # Broadcasting works row-wise and we then check any full-row match
        mask = (done.values == values).all(axis=1)
        return mask.any()

    remaining_scalar_dicts = [p for p in all_param_dicts if not is_done(p)]

    # Convert remaining combinations to "small grids" acceptable by GridSearchCV
    remaining_grids = [{k: [v] for k, v in p.items()} for p in remaining_scalar_dicts]

    return remaining_grids

###############
# Model constructor wrapper for SciKeras
###############
def create_model_for_cv(
    input_length: int,
    target_class: int,
    num_layers: int = int(ADVANCED_DEFAULT_CONFIG.get("num_layers", 3)),
    dense_units: int = int(ADVANCED_DEFAULT_CONFIG.get("dense_units", 200)),
    dropout: float = float(ADVANCED_DEFAULT_CONFIG.get("dropout", 0.3)),
    optimizer: str = str(ADVANCED_DEFAULT_CONFIG.get("optimizer", "adam")),
    learning_rate: float = float(ADVANCED_DEFAULT_CONFIG.get("learning_rate", 1e-3)),
    use_batchnorm: bool = bool(ADVANCED_DEFAULT_CONFIG.get("use_batchnorm", True)),
    activation: str = str(ADVANCED_DEFAULT_CONFIG.get("activation", "gelu")),
    ridge_penalty : float = float(ADVANCED_DEFAULT_CONFIG.get("ridge_penalty", 0.0)),
    lasso_penalty : float = float(ADVANCED_DEFAULT_CONFIG.get("lasso_penalty", 0.0))
):
    """
    SciKeras-compatible factory.

    Parameters are passed automatically by GridSearchCV according to
    ADVANCED_PARAM_GRID. Defaults are taken from ADVANCED_DEFAULT_CONFIG.
    """
    return create_model(
        input_length=input_length,
        target_class=target_class,
        num_layers=num_layers,
        dense_units=dense_units,
        dropout=dropout,
        optimizer=optimizer,
        learning_rate=learning_rate,
        use_batchnorm=use_batchnorm,
        activation=activation,
        ridge_penalty =ridge_penalty,
        lasso_penalty=lasso_penalty
    )

###############
# CV Routine
###############
def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Sequence[Mapping[str, Any]],
    cv_splits: int,
    seed: int,
    target_class: int,
    input_length: int,
):
    """
    Cross-validation with standardization and hyperparameter tuning
    for the advanced model.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1,
    )

    lr_plateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    ### Build SciKeras classifier inside a sklearn Pipeline
    ## StandardScaler -> KerasClassifier
    keras_clf = KerasClassifier(
        model=create_model_for_cv,
        input_length=input_length,
        target_class=target_class,
        verbose=1,
        random_state=seed,
        loss="sparse_categorical_crossentropy",
        validation_split=0.2,
        callbacks=[early_stopping, lr_plateau],
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", keras_clf),
        ]
    )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=[dict(p) for p in param_grid],
        cv=cv,
        scoring=scoring,
        refit="accuracy",
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )

    grid.fit(X_train, y_train)
    return grid


###############
# Helpers: store results, retrain best, evaluate
###############
def results_to_dataframe(grid, out_csv_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_accuracy")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_csv_path.exists()
    df.to_csv(
        out_csv_path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )
    print(f"[saved] {out_csv_path}")
    return df


def train_best_and_save(
    grid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: Path,
):
    """
    Refit best pipeline on full training data and save the underlying Keras model.
    """
    best_pipeline = grid.best_estimator_
    best_pipeline.fit(X_train, y_train)

    best_model = best_pipeline.named_steps["clf"].model_
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(model_path)
    print(f"[saved] {model_path}")

    return best_pipeline, best_model


def evaluate_on_test(
    best_pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute test accuracy, F1 macro and log-loss on the held-out test set.
    """
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    ll = log_loss(y_test, y_proba)

    return {"accuracy": acc, "f1_macro": f1, "log_loss": ll}


###############
# High-level CV runner
###############
def run_cross_validation(
    training_dataset: Path,
    cv_splits: int,
    seed: int,
    results_csv: Path = RESULTS_DIR / "results_advanced_with_L1_L2.csv",
) -> None:
    ### Load and split data
    ## Balanced split using shared utility
    emotions = EMOTIONS_ADVANCED

    int2emotions, emotions2int, x_train, y_train, x_test, y_test, _, _ = split_data(
        emotions=emotions,
        filename=training_dataset,
        train_set=0.8,
        seed=seed,
    )

    y_train = np.array([emotions2int[label] for label in y_train.ravel()])
    y_test  = np.array([emotions2int[label] for label in y_test.ravel()])

    target_class = len(emotions)
    input_length = x_train.shape[1]

    print("Running grid search for ADVANCED model...")
    print(f"Training dataset : {training_dataset}")
    print(f"Input length     : {input_length}")
    print(f"Number of classes: {target_class}")
    print(f"CV splits        : {cv_splits}, seed={seed}")

    filtered_param_grid = build_filtered_param_grid(results_csv)

    if not filtered_param_grid:
        print("All hyperparameter combinations have already been evaluated.")
        return
    
    ### Run grid search
    grid = run_grid_search(
        X_train=x_train,
        y_train=y_train,
        param_grid=filtered_param_grid,
        cv_splits=cv_splits,
        seed=seed,
        target_class=target_class,
        input_length=input_length,
    )

    ### Save CV results
    _ = results_to_dataframe(grid, results_csv)

    ### Train best model and save
    best_model_path = RESULTS_DIR / "advanced_best_model_in_CV.keras"
    best_pipeline, _ = train_best_and_save(
        grid=grid,
        X_train=x_train,
        y_train=y_train,
        model_path=best_model_path,
    )

    ### Evaluate on held-out test set
    metrics = evaluate_on_test(
        best_pipeline=best_pipeline,
        X_test=x_test,
        y_test=y_test,
    )

    print("Test metrics:", metrics)
    print(f"Best model saved to: {best_model_path}")
    print(f"CV results saved to: {results_csv}")


###############
# CLI
###############
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validation + hyperparameter search for the advanced model."
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=ROOT / "data_training" / "merged_with_features.csv",
        help="CSV with training features (default: data_training/merged_with_features.csv).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and CV (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    set_global_seed()
    args = _parse_args()
    run_cross_validation(
        training_dataset=args.training_dataset,
        cv_splits=args.cv_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
