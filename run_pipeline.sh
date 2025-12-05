#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Global config
###############################################################################

# Detect repository root (directory of this script)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

PYTHON=python

# Optionally load HF_TOKEN and others from .env (HF_TOKEN=hf_xxx...)
if [ -f "$ROOT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +o allexport
  echo "[INFO] Loaded environment variables from .env"
fi

###############################################################################
# Paths – training data
###############################################################################

TRAIN_RAW_DIR="$ROOT_DIR/data_training/raw_data"
TRAIN_CONVERT_DIR="$ROOT_DIR/data_training/converted_16khz"
TRAIN_MANIFEST="$ROOT_DIR/data_training/merged.csv"
TRAIN_FEATURE_BASE="$ROOT_DIR/data_training/merged_with_features.csv"

TRAIN_FEATURE_LIBROSA="${TRAIN_FEATURE_BASE%.*}_librosa.csv"
TRAIN_FEATURE_OPENSMILE="${TRAIN_FEATURE_BASE%.*}_opensmile.csv"
TRAIN_FEATURE_BOTH="$TRAIN_FEATURE_BASE"

###############################################################################
# Paths – FOMC data
###############################################################################

FOMC_DOWNLOAD_DIR="$ROOT_DIR/data_fomc/raw_data"
FOMC_CONVERT_DIR="$ROOT_DIR/data_fomc/converted_16khz"
FOMC_FEATURE_DIR="$ROOT_DIR/data_fomc/features"

FOMC_FEATURE_LIBROSA="$FOMC_FEATURE_DIR/features_librosa.parquet"
FOMC_FEATURE_OPENSMILE="$FOMC_FEATURE_DIR/features_opensmile.parquet"
FOMC_FEATURE_MERGED="$FOMC_FEATURE_DIR/features_merged.parquet"

###############################################################################
# Paths – models, experiments, outputs
###############################################################################

OUTPUT_DIR="$ROOT_DIR/outputs"
EXPERIMENTS_DIR="$OUTPUT_DIR/experiments"
EDA_DIR="$OUTPUT_DIR/eda_analysis"
PLOTS_DIR="$EXPERIMENTS_DIR/plots"

MODEL_DIR="$OUTPUT_DIR/models"
PRED_DIR="$OUTPUT_DIR/predictions"

ADV_MODEL_PATH="$MODEL_DIR/advanced_model.h5"
ADV_METRICS_CSV="$MODEL_DIR/advanced_metrics.csv"
ADV_FOMC_PRED="$PRED_DIR/advanced_fomc_predictions.csv"

CV_RESULTS_CSV="$EXPERIMENTS_DIR/results_advanced.csv"

mkdir -p "$TRAIN_RAW_DIR" "$TRAIN_CONVERT_DIR"
mkdir -p "$FOMC_DOWNLOAD_DIR" "$FOMC_CONVERT_DIR" "$FOMC_FEATURE_DIR"
mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$PRED_DIR"

###############################################################################
# Hyperparameters for final ADVANCED model (from best CV config)
###############################################################################
ADV_NUM_LAYERS=4
ADV_DENSE_UNITS=256
ADV_DROPOUT=0.2
ADV_OPTIMIZER="rmsprop"
ADV_LR=0.001
ADV_BATCH_SIZE=32
ADV_EPOCHS=100
ADV_ACTIVATION="gelu"
ADV_USE_BATCHNORM=1   # boolean flag
# ridge_penalty = 0.0 (relies on model/config default being 0.0)

###############################################################################
# 1) TRAINING DATA – download + manifest + feature extraction (librosa+opensmile)
###############################################################################
echo
echo "================================================================"
echo "[1/8] Training data: download + assemble + features (librosa+opensmile)"
echo "================================================================"

# This script downloads Kaggle datasets (unless --no-download) and then:
#  - builds a merged manifest
#  - extracts Librosa + OpenSMILE features
#  - writes:
#       TRAIN_FEATURE_LIBROSA
#       TRAIN_FEATURE_OPENSMILE
#       TRAIN_FEATURE_BOTH
$PYTHON helpers/run_extract_training_features.py \
  --raw-data-dir "$TRAIN_RAW_DIR" \
  --manifest "$TRAIN_MANIFEST" \
  --out-file "$TRAIN_FEATURE_BASE" \
  --conversion-dir "$TRAIN_CONVERT_DIR" \
  --engine both \
  --emotions 1 2 3 4 5 6 7 8

echo "[INFO] Training feature tables:"
echo "   Librosa   : $TRAIN_FEATURE_LIBROSA"
echo "   OpenSMILE : $TRAIN_FEATURE_OPENSMILE"
echo "   Combined  : $TRAIN_FEATURE_BOTH"

###############################################################################
# 2) FOMC DATA – download + convert + features (librosa+opensmile)
###############################################################################
echo
echo "================================================================"
echo "[2/8] FOMC data: download + convert + features (librosa+opensmile)"
echo "================================================================"

$PYTHON helpers/run_extract_fomc_features.py \
  --download-dir "$FOMC_DOWNLOAD_DIR" \
  --convert-dir  "$FOMC_CONVERT_DIR" \
  --features-dir "$FOMC_FEATURE_DIR" \
  --engine both

echo "[INFO] FOMC feature tables:"
echo "   Librosa   : $FOMC_FEATURE_LIBROSA"
echo "   OpenSMILE : $FOMC_FEATURE_OPENSMILE"
echo "   Combined  : $FOMC_FEATURE_MERGED"

###############################################################################
# 3) EDA on training features (Librosa + OpenSMILE)
###############################################################################
echo
echo "================================================================"
echo "[3/8] EDA on training features"
echo "================================================================"

$PYTHON helpers/run_eda_analysis.py \
  --librosa-csv "$TRAIN_FEATURE_LIBROSA" \
  --opensmile-csv "$TRAIN_FEATURE_OPENSMILE" \
  --save-dir "$EDA_DIR"

echo "[INFO] EDA outputs stored in: $EDA_DIR"

###############################################################################
# 4) Cross-validation + hyperparameter search (ADVANCED model)
###############################################################################
echo
echo "================================================================"
echo "[4/8] Cross-validation + hyperparameter search (ADVANCED)"
echo "================================================================"

mkdir -p "$EXPERIMENTS_DIR"

$PYTHON helpers/run_cross_validation.py \
  --training-dataset "$TRAIN_FEATURE_BOTH" \
  --cv-splits 5 \
  --seed 42

echo "[INFO] CV results written to: $CV_RESULTS_CSV"

###############################################################################
# 5) Visualization of CV / hyperparameter search
###############################################################################
echo
echo "================================================================"
echo "[5/8] Visualization of hyperparameter search / CV"
echo "================================================================"

$PYTHON helpers/run_visualisation.py \
  --results-csv "$CV_RESULTS_CSV" \
  --save-dir "$PLOTS_DIR" \
  all

echo "[INFO] Visualization plots stored in: $PLOTS_DIR"

###############################################################################
# 6) Train final ADVANCED model with selected hyperparameters
###############################################################################
echo
echo "================================================================"
echo "[6/8] Train final ADVANCED model with best hyperparameters"
echo "================================================================"

$PYTHON helpers/run_training_model.py \
  --mode advanced \
  --infile "$TRAIN_FEATURE_BOTH" \
  --model-path "$ADV_MODEL_PATH" \
  --metrics-csv "$ADV_METRICS_CSV" \
  --num-layers "$ADV_NUM_LAYERS" \
  --dense-units "$ADV_DENSE_UNITS" \
  --dropout "$ADV_DROPOUT" \
  --optimizer "$ADV_OPTIMIZER" \
  --learning-rate "$ADV_LR" \
  --activation "$ADV_ACTIVATION" \
  --batch-size "$ADV_BATCH_SIZE" \
  --epochs "$ADV_EPOCHS" \
  --use-batchnorm \
  --standardize-inputs

echo "[INFO] Trained ADVANCED model: $ADV_MODEL_PATH"
echo "[INFO] Metrics CSV           : $ADV_METRICS_CSV"

###############################################################################
# 7) Test / inference on FOMC features with final ADVANCED model
###############################################################################
echo
echo "================================================================"
echo "[7/8] Predict emotions on FOMC features (ADVANCED model)"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode advanced \
  --weights "$ADV_MODEL_PATH" \
  --infile "$FOMC_FEATURE_MERGED" \
  --outfile "$ADV_FOMC_PRED"

echo "[INFO] FOMC predictions stored in: $ADV_FOMC_PRED"

###############################################################################
# 8) Summary
###############################################################################
echo
echo "================================================================"
echo "[8/8] Pipeline completed"
echo "================================================================"
echo "Training features : $TRAIN_FEATURE_BOTH"
echo "FOMC features     : $FOMC_FEATURE_MERGED"
echo "Model weights     : $ADV_MODEL_PATH"
echo "Metrics           : $ADV_METRICS_CSV"
echo "FOMC predictions  : $ADV_FOMC_PRED"
echo "================================================================"
