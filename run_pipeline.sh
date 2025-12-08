#!/usr/bin/env bash
export TF_ENABLE_ONEDNN_OPTS=0
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

# Assuming manifest is a single file inside "merged"
TRAIN_MERGED_DIR="$ROOT_DIR/data_training/merged"
TRAIN_MANIFEST="$TRAIN_MERGED_DIR/manifest"

TRAIN_FEATURE_BASE="$ROOT_DIR/data_training/merged_with_features.csv"

TRAIN_FEATURE_LIBROSA="${TRAIN_FEATURE_BASE%.*}_librosa.csv"
TRAIN_FEATURE_OPENSMILE="${TRAIN_FEATURE_BASE%.*}_opensmile.csv"
TRAIN_FEATURE_BOTH="$TRAIN_FEATURE_BASE"

# Filtered training features for ravdess+tess (file)
TRAIN_FEATURE_LIBROSA_FILTERED="${TRAIN_FEATURE_LIBROSA%.*}_filtered_ravdess_tess.csv"

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

## General folder
OUTPUT_DIR="$ROOT_DIR/outputs"

## Secondary folders
EXPERIMENTS_DIR="$OUTPUT_DIR/experiments"
EDA_DIR="$OUTPUT_DIR/eda_analysis"
BASELINE_MODEL_DIR="$OUTPUT_DIR/baseline_model"
FINAL_MODEL_DIR="$OUTPUT_DIR/final_model"

### Subdirs
#### Experiments
PLOTS_DIR="$EXPERIMENTS_DIR/plots"
CSV_RESULTS_OF_CV_PLUS_GS="$EXPERIMENTS_DIR/results_advanced_with_L1_L2.csv"
KERAS_BEST_MODEL_OF_CV_PLUS_GS="$EXPERIMENTS_DIR/advanced_best_model_in_CV.keras"

#### Eda analysis
EMOTIONS_DISTRIBUTION_DIR="$EDA_DIR/emotion_distribution"
LIBROSA_FEATURE_RELATIONSHIPS_DIR="$EDA_DIR/librosa_feature_relationships"
OPENSMILE_FEATURE_RELATIONSHIPS_DIR="$EDA_DIR/opensmile_feature_relationships"
TSNE_DIR="$EDA_DIR/tsne"

#### Baseline model
# BOTH_BASELINE_MODEL_DIR="$BASELINE_MODEL_DIR/both"
LIBROSA_BASELINE_MODEL_DIR="$BASELINE_MODEL_DIR/librosa"
KERAS_BASELINE_MODEL="$BASELINE_MODEL_DIR/librosa_only_model.keras"
CSV_METRICS_BASELINE_MODEL="$BASELINE_MODEL_DIR/librosa_only_metrics.csv"
# OPENSMILE_BASELINE_MODEL_DIR="$BASELINE_MODEL_DIR/opensmile"

#### Final model
ALL_DATASETS_FINAL_MODEL_DIR="$FINAL_MODEL_DIR/all_datasets"
KERAS_ADVANCED_MODEL="$ALL_DATASETS_FINAL_MODEL_DIR/final_model.keras"
CSV_METRICS_ADVANCED_MODEL="$ALL_DATASETS_FINAL_MODEL_DIR/final_model_metrics.csv"
# RAVDESS_TESS_DATASETS_FINAL_MODEL_DIR="$FINAL_MODEL_DIR/ravdess_tess"

#### Prediction files
LIBROSA_FOMC_PRED="$LIBROSA_BASELINE_MODEL_DIR/predictions_fomc.csv"
ADV_FOMC_PRED="$ALL_DATASETS_FINAL_MODEL_DIR/predictions_fomc.csv"
LIBROSA_TRAIN_PRED="$LIBROSA_BASELINE_MODEL_DIR/predictions_training.csv"
ADV_TRAIN_PRED="$ALL_DATASETS_FINAL_MODEL_DIR/predictions_training.csv"

###############################################################################
# Creating folders
###############################################################################

mkdir -p \
  "$TRAIN_RAW_DIR" \
  "$TRAIN_CONVERT_DIR" \
  "$TRAIN_MERGED_DIR" \
  \
  "$FOMC_DOWNLOAD_DIR" \
  "$FOMC_CONVERT_DIR" \
  "$FOMC_FEATURE_DIR" \
  \
  "$OUTPUT_DIR" \
  "$EXPERIMENTS_DIR" \
  "$EDA_DIR" \
  "$BASELINE_MODEL_DIR" \
  "$FINAL_MODEL_DIR" \
  \
  "$PLOTS_DIR" \
  \
  "$EMOTIONS_DISTRIBUTION_DIR" \
  "$LIBROSA_FEATURE_RELATIONSHIPS_DIR" \
  "$OPENSMILE_FEATURE_RELATIONSHIPS_DIR" \
  "$TSNE_DIR" \
  \
  "$LIBROSA_BASELINE_MODEL_DIR" \
  "$ALL_DATASETS_FINAL_MODEL_DIR"

###############################################################################
# Hyperparameters for final ADVANCED model
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
# 1) DOWNLOAD TRAINING DATA – download + manifest + feature extraction (librosa+opensmile)
###############################################################################
echo
echo "================================================================"
echo "[1/8] Training data: download + assemble + features (librosa+opensmile)"
echo "================================================================"

$PYTHON helpers/run_extract_training_features.py \
  --raw-data-dir "$TRAIN_RAW_DIR" \
  --manifest "$TRAIN_MANIFEST" \
  --out-file "$TRAIN_FEATURE_BASE" \
  --conversion-dir "$TRAIN_CONVERT_DIR" \
  --engine both \
  --emotions 1 2 3 4 5 6 7 8 9

echo "[INFO] Training feature tables:"
echo "   Librosa   : $TRAIN_FEATURE_LIBROSA"
echo "   OpenSMILE : $TRAIN_FEATURE_OPENSMILE"
echo "   Both      : $TRAIN_FEATURE_BOTH"

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

echo "[INFO] CV results written to: $CSV_RESULTS_OF_CV_PLUS_GS"

###############################################################################
# 5) Training models: Baseline(librosa), ADVANCED(full datasets)
###############################################################################
echo
echo "================================================================"
echo "[5/8] Training models"
echo "================================================================"

echo
echo "================================================================"
echo "[5.1/8] Training BASELINE model"
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
echo "[5.2/8] Training ADVANCED model"
echo "================================================================"

$PYTHON helpers/run_training_model.py \
  --mode advanced \
  --infile "$TRAIN_FEATURE_BOTH" \
  --model-path "$KERAS_ADVANCED_MODEL" \
  --metrics-csv "$CSV_METRICS_ADVANCED_MODEL" \
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

echo "[INFO] Trained ADVANCED model: $KERAS_ADVANCED_MODEL"
echo "[INFO] Metrics CSV            : $CSV_METRICS_ADVANCED_MODEL"

###############################################################################
# 6) Prediction on FOMC Dataset
###############################################################################
echo
echo "================================================================"
echo "[6/8] Predicting FOMC emotions"
echo "================================================================"

echo
echo "================================================================"
echo "[6.1/8] Predict emotions on FOMC features BASELINE model"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode baseline \
  --weights "$KERAS_BASELINE_MODEL" \
  --infile "$FOMC_FEATURE_LIBROSA" \
  --outfile "$LIBROSA_FOMC_PRED"

echo "[INFO] FOMC predictions stored in: $LIBROSA_FOMC_PRED"

echo
echo "================================================================"
echo "[6.2/8] Predict emotions on FOMC features ADVANCED model"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode advanced \
  --weights "$KERAS_ADVANCED_MODEL" \
  --infile "$FOMC_FEATURE_MERGED" \
  --outfile "$ADV_FOMC_PRED"

echo "[INFO] FOMC predictions stored in: $ADV_FOMC_PRED"

###############################################################################
# 7) Prediction on Training Dataset
###############################################################################
echo
echo "================================================================"
echo "[7/8] Predicting TRAINING emotions"
echo "================================================================"

echo
echo "================================================================"
echo "[7.1/8] Predict emotions on TRAINING features BASELINE model"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode baseline \
  --weights "$KERAS_BASELINE_MODEL" \
  --infile "$TRAIN_FEATURE_LIBROSA_FILTERED" \
  --outfile "$LIBROSA_TRAIN_PRED"

echo "[INFO] Training predictions stored in: $LIBROSA_TRAIN_PRED"

echo
echo "================================================================"
echo "[7.2/8] Predict emotions on TRAINING features ADVANCED model"
echo "================================================================"

$PYTHON helpers/run_predict_model.py \
  --mode advanced \
  --weights "$KERAS_ADVANCED_MODEL" \
  --infile "$TRAIN_FEATURE_BASE" \
  --outfile "$ADV_TRAIN_PRED"

echo "[INFO] Training predictions stored in: $ADV_TRAIN_PRED"

###############################################################################
# 8) Summary
###############################################################################
echo
echo "================================================================"
echo "[8/8] Pipeline completed"
echo "================================================================"
echo "Training features                 : $TRAIN_FEATURE_BOTH"
echo "FOMC features                     : $FOMC_FEATURE_MERGED"
echo "Final model weights               : $KERAS_ADVANCED_MODEL"
echo "Baseline model weights            : $KERAS_BASELINE_MODEL"
echo "Final model metrics               : $CSV_METRICS_ADVANCED_MODEL"
echo "Baseline model metrics            : $CSV_METRICS_BASELINE_MODEL"
echo "Final model FOMC predictions      : $ADV_FOMC_PRED"
echo "Baseline model FOMC predictions   : $LIBROSA_FOMC_PRED"
echo "Final model TRAINING predictions  : $ADV_TRAIN_PRED"
echo "Baseline TRAINING predictions     : $LIBROSA_TRAIN_PRED"
echo "================================================================"

echo "================================================================"
echo "FINAL MODEL ARCHITECTURE"
echo "================================================================"
echo "Number of layers       : $ADV_NUM_LAYERS"
echo "Number of dense units  : $ADV_DENSE_UNITS"
echo "Dropout rate           : $ADV_DROPOUT"
echo "Optimizer              : $ADV_OPTIMIZER"
echo "Starting learning rate : $ADV_LR"
echo "Batch size             : $ADV_BATCH_SIZE"
echo "Activation function    : $ADV_ACTIVATION"
echo "Ridge penalty          : $ADV_RIDGE_PENALTY"
echo "Lasso penalty          : $ADV_LASSO_PENALTY"
echo "Inputs are normalized and batchnorm is used"
