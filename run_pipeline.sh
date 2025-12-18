#!/usr/bin/env bash
export TF_ENABLE_ONEDNN_OPTS=0
set -euo pipefail

###############################################################################
# Global config
###############################################################################

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

PYTHON=python

if [ -f "$ROOT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +o allexport
  echo "[INFO] Loaded environment variables from .env"
fi

###############################################################################
# Paths – training feature files (ASSUMED TO EXIST)
###############################################################################

TRAIN_FEATURE_BASE="$ROOT_DIR/data_training/merged_with_features.csv"
TRAIN_FEATURE_LIBROSA="${TRAIN_FEATURE_BASE%.*}_librosa.csv"
TRAIN_FEATURE_BOTH="$TRAIN_FEATURE_BASE"

# This may be created by the training script when using --datasets ravdess tess
TRAIN_FEATURE_LIBROSA_FILTERED="${TRAIN_FEATURE_LIBROSA%.*}_filtered_ravdess_tess.csv"

###############################################################################
# Paths – outputs (models, metrics, predictions)
###############################################################################

OUTPUT_DIR="$ROOT_DIR/outputs"
BASELINE_MODEL_DIR="$OUTPUT_DIR/baseline_model"
FINAL_MODEL_DIR="$OUTPUT_DIR/final_model"

LIBROSA_BASELINE_MODEL_DIR="$BASELINE_MODEL_DIR/librosa"
KERAS_BASELINE_MODEL="$BASELINE_MODEL_DIR/librosa_only_model.keras"
CSV_METRICS_BASELINE_MODEL="$BASELINE_MODEL_DIR/librosa_only_metrics.csv"

ALL_DATASETS_FINAL_MODEL_DIR="$FINAL_MODEL_DIR/all_datasets"
ALL_DATASETS_KERAS_ADVANCED_MODEL="$ALL_DATASETS_FINAL_MODEL_DIR/final_model.keras"
ALL_DATASETS_CSV_METRICS_ADVANCED_MODEL="$ALL_DATASETS_FINAL_MODEL_DIR/final_model_metrics.csv"

RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR="$FINAL_MODEL_DIR/ravdess_tess"
RAVDESS_TESS_KERAS_ADVANCED_MODEL="$RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR/final_model.keras"
RAVDESS_TESS_CSV_METRICS_ADVANCED_MODEL="$RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR/final_model_metrics.csv"

LIBROSA_TRAIN_PRED="$LIBROSA_BASELINE_MODEL_DIR/predictions_training.csv"
ALL_DATASETS_ADV_TRAIN_PRED="$ALL_DATASETS_FINAL_MODEL_DIR/predictions_training.csv"
RAVDESS_TESS_ADV_TRAIN_PRED="$RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR/predictions_training.csv"

###############################################################################
# Creating folders
###############################################################################

mkdir -p \
  "$OUTPUT_DIR" \
  "$BASELINE_MODEL_DIR" \
  "$FINAL_MODEL_DIR" \
  "$LIBROSA_BASELINE_MODEL_DIR" \
  "$ALL_DATASETS_FINAL_MODEL_DIR" \
  "$RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR"

###############################################################################
# Hyperparameters for final ADVANCED model (used for BOTH advanced trainings)
###############################################################################
ADV_NUM_LAYERS=2
ADV_DENSE_UNITS=512
ADV_DROPOUT=0.1
ADV_OPTIMIZER="adam"
ADV_LR=0.001
ADV_BATCH_SIZE=128
ADV_EPOCHS=300
ADV_ACTIVATION="gelu"
ADV_RIDGE_PENALTY=0.0
ADV_LASSO_PENALTY=1e-8

###############################################################################
# Hyperparameters for BASELINE model
###############################################################################
BAS_NUM_LAYERS=3
BAS_DENSE_UNITS=200
BAS_DROPOUT=0.3
BAS_OPTIMIZER="adam"
BAS_LR=0.001
BAS_BATCH_SIZE=64
BAS_EPOCHS=1000
BAS_ACTIVATION="linear"
BAS_RIDGE_PENALTY=0.0
BAS_LASSO_PENALTY=0.0

###############################################################################
# Sanity checks: only the base feature files must exist
###############################################################################

echo
echo "================================================================"
echo "[0/3] Checking required feature files exist"
echo "================================================================"

for f in "$TRAIN_FEATURE_LIBROSA" "$TRAIN_FEATURE_BOTH"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing required file: $f"
    exit 1
  fi
done

echo "[INFO] Found:"
echo "   Librosa features           : $TRAIN_FEATURE_LIBROSA"
echo "   Both-engine merged features: $TRAIN_FEATURE_BOTH"
echo "[INFO] Filtered librosa file may be created later:"
echo "   $TRAIN_FEATURE_LIBROSA_FILTERED"

###############################################################################
# 1) Train models (this step is expected to create the filtered file)
###############################################################################
echo
echo "================================================================"
echo "[1/3] Training models"
echo "================================================================"

echo
echo "================================================================"
echo "[1.1/3] Training BASELINE model (RAVDESS+TESS)"
echo "================================================================"

$PYTHON helpers/run_training_model.py \
  --mode baseline \
  --infile "$TRAIN_FEATURE_LIBROSA" \
  --model-path "$KERAS_BASELINE_MODEL" \
  --metrics-csv "$CSV_METRICS_BASELINE_MODEL" \
  --num-layers "$BAS_NUM_LAYERS" \
  --dense-units "$BAS_DENSE_UNITS" \
  --dropout "$BAS_DROPOUT" \
  --optimizer "$BAS_OPTIMIZER" \
  --learning-rate "$BAS_LR" \
  --activation "$BAS_ACTIVATION" \
  --batch-size "$BAS_BATCH_SIZE" \
  --epochs "$BAS_EPOCHS" \
  --datasets ravdess tess

echo "[INFO] Trained BASELINE model: $KERAS_BASELINE_MODEL"
echo "[INFO] Metrics CSV            : $CSV_METRICS_BASELINE_MODEL"

echo
echo "================================================================"
echo "[1.2/3] Training ADVANCED model (ALL datasets)"
echo "================================================================"

$PYTHON helpers/run_training_model.py \
  --mode advanced \
  --infile "$TRAIN_FEATURE_BOTH" \
  --model-path "$ALL_DATASETS_KERAS_ADVANCED_MODEL" \
  --metrics-csv "$ALL_DATASETS_CSV_METRICS_ADVANCED_MODEL" \
  --num-layers "$ADV_NUM_LAYERS" \
  --dense-units "$ADV_DENSE_UNITS" \
  --dropout "$ADV_DROPOUT" \
  --optimizer "$ADV_OPTIMIZER" \
  --learning-rate "$ADV_LR" \
  --activation "$ADV_ACTIVATION" \
  --batch-size "$ADV_BATCH_SIZE" \
  --epochs "$ADV_EPOCHS" \
  --ridge-penalty "$ADV_RIDGE_PENALTY" \
  --lasso-penalty "$ADV_LASSO_PENALTY" \
  --use-batchnorm \
  --standardize-inputs

echo "[INFO] Trained ADVANCED model (ALL) : $ALL_DATASETS_KERAS_ADVANCED_MODEL"
echo "[INFO] Metrics CSV (ALL)            : $ALL_DATASETS_CSV_METRICS_ADVANCED_MODEL"

echo
echo "================================================================"
echo "[1.3/3] Training ADVANCED model (RAVDESS + TESS subsample)"
echo "================================================================"

$PYTHON helpers/run_training_model.py \
  --mode advanced \
  --infile "$TRAIN_FEATURE_BOTH" \
  --model-path "$RAVDESS_TESS_KERAS_ADVANCED_MODEL" \
  --metrics-csv "$RAVDESS_TESS_CSV_METRICS_ADVANCED_MODEL" \
  --num-layers "$ADV_NUM_LAYERS" \
  --dense-units "$ADV_DENSE_UNITS" \
  --dropout "$ADV_DROPOUT" \
  --optimizer "$ADV_OPTIMIZER" \
  --learning-rate "$ADV_LR" \
  --activation "$ADV_ACTIVATION" \
  --batch-size "$ADV_BATCH_SIZE" \
  --epochs "$ADV_EPOCHS" \
  --ridge-penalty "$ADV_RIDGE_PENALTY" \
  --lasso-penalty "$ADV_LASSO_PENALTY" \
  --use-batchnorm \
  --standardize-inputs \
  --datasets ravdess tess

echo "[INFO] Trained ADVANCED model (R+T): $RAVDESS_TESS_KERAS_ADVANCED_MODEL"
echo "[INFO] Metrics CSV (R+T)           : $RAVDESS_TESS_CSV_METRICS_ADVANCED_MODEL"

###############################################################################
# 2) Predict on TRAINING dataset (no FOMC)
###############################################################################
echo
echo "================================================================"
echo "[2/3] Predicting TRAINING emotions (no FOMC)"
echo "================================================================"

# Baseline training predictions: prefer filtered file if it exists; else fall back to full librosa set
BASELINE_PRED_INFILE="$TRAIN_FEATURE_LIBROSA"
if [ -f "$TRAIN_FEATURE_LIBROSA_FILTERED" ]; then
  BASELINE_PRED_INFILE="$TRAIN_FEATURE_LIBROSA_FILTERED"
  echo "[INFO] Using filtered librosa file for baseline predictions: $BASELINE_PRED_INFILE"
else
  echo "[WARN] Filtered librosa file not found, using full librosa features instead: $BASELINE_PRED_INFILE"
fi

echo
echo "================================================================"
echo "[2.1/3] Predict emotions on TRAINING features BASELINE model"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode baseline \
  --weights "$KERAS_BASELINE_MODEL" \
  --infile "$BASELINE_PRED_INFILE" \
  --outfile "$LIBROSA_TRAIN_PRED"

echo "[INFO] Training predictions stored in: $LIBROSA_TRAIN_PRED"

echo
echo "================================================================"
echo "[2.2/3] Predict emotions on TRAINING features ADVANCED model (ALL datasets)"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode advanced \
  --weights "$ALL_DATASETS_KERAS_ADVANCED_MODEL" \
  --infile "$TRAIN_FEATURE_BASE" \
  --outfile "$ALL_DATASETS_ADV_TRAIN_PRED"

echo "[INFO] Training predictions stored in: $ALL_DATASETS_ADV_TRAIN_PRED"

echo
echo "================================================================"
echo "[2.3/3] Predict emotions on TRAINING features ADVANCED model (RAVDESS + TESS)"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode advanced \
  --weights "$RAVDESS_TESS_KERAS_ADVANCED_MODEL" \
  --infile "$TRAIN_FEATURE_BASE" \
  --outfile "$RAVDESS_TESS_ADV_TRAIN_PRED"

echo "[INFO] Training predictions stored in: $RAVDESS_TESS_ADV_TRAIN_PRED"

###############################################################################
# 3) Summary
###############################################################################
echo
echo "================================================================"
echo "[3/3] Pipeline completed (training + training-set predictions only)"
echo "================================================================"
echo "Baseline model weights                 : $KERAS_BASELINE_MODEL"
echo "Advanced model weights (ALL)           : $ALL_DATASETS_KERAS_ADVANCED_MODEL"
echo "Advanced model weights (RAVDESS+TESS)  : $RAVDESS_TESS_KERAS_ADVANCED_MODEL"
echo "Baseline model metrics                 : $CSV_METRICS_BASELINE_MODEL"
echo "Advanced model metrics (ALL)           : $ALL_DATASETS_CSV_METRICS_ADVANCED_MODEL"
echo "Advanced model metrics (RAVDESS+TESS)  : $RAVDESS_TESS_CSV_METRICS_ADVANCED_MODEL"
echo "Baseline model TRAIN predictions       : $LIBROSA_TRAIN_PRED"
echo "Advanced model TRAIN predictions (ALL) : $ALL_DATASETS_ADV_TRAIN_PRED"
echo "Advanced model TRAIN predictions (R+T) : $RAVDESS_TESS_ADV_TRAIN_PRED"
echo "================================================================"
