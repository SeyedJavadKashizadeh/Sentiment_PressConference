#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Global config
###############################################################################

# Detect repository root (directory of this script)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

PYTHON=python

# Optionally load HF_TOKEN from .env (HF_TOKEN=hf_xxx...)
if [ -f "$ROOT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
  set +o allexport
  echo "[INFO] Loaded environment variables from .env"
fi

###############################################################################
# Paths for TRAINING DATA
###############################################################################

TRAIN_RAW_DIR="$ROOT_DIR/data_training/raw_data"
TRAIN_CONVERTED_DIR="$ROOT_DIR/data_training/converted_16khz"
TRAIN_MANIFEST="$ROOT_DIR/data_training/merged"
TRAIN_FEATURE_BASE="$ROOT_DIR/data_training/merged_with_features.csv"

# When engine=both, we assume:
TRAIN_FEATURE_LIBROSA="${TRAIN_FEATURE_BASE%.*}_librosa.csv"
TRAIN_FEATURE_OPENSMILE="${TRAIN_FEATURE_BASE%.*}_opensmile.csv"
TRAIN_FEATURE_BOTH="$TRAIN_FEATURE_BASE"

###############################################################################
# Paths for FOMC DATA
###############################################################################

FOMC_DOWNLOAD_DIR="$ROOT_DIR/data_fomc/raw_data"
FOMC_CONVERT_DIR="$ROOT_DIR/data_fomc/converted_16khz"
FOMC_FEATURE_DIR="$ROOT_DIR/data_fomc/features"

###############################################################################
# Paths for model outputs
###############################################################################

OUTPUT_LIBROSA="$ROOT_DIR/outputs/librosa"
OUTPUT_OPENSMILE="$ROOT_DIR/outputs/opensmile"
OUTPUT_COMBINED="$ROOT_DIR/outputs/both"

###############################################################################
# 1) TRAINING PIPELINE:
#    download training datasets, assemble manifest, extract features (both engines)
###############################################################################

echo
echo "================================================================"
echo "[1/5] Preparing TRAINING data (download + assemble + features)"
echo "================================================================"

# Example: gender + emotions filtering — adjust as you like.
# If your run_extract_training_features.py signature is different, adapt this call.
$PYTHON helpers/run_extract_training_features.py \
  --emotions 1 2 3 4 5 6 7 8 \
  --manifest "$TRAIN_MANIFEST" \
  --out-file "$TRAIN_FEATURE_BASE" \
  --engine both

echo "[INFO] Expected training feature tables:"
echo "   Librosa   : $TRAIN_FEATURE_LIBROSA"
echo "   OpenSMILE : $TRAIN_FEATURE_OPENSMILE"
echo "   Combined  : $TRAIN_FEATURE_BOTH"

###############################################################################
# 2) FOMC PIPELINE:
#    download FOMC audios + extract features (both engines)
###############################################################################

echo
echo "================================================================"
echo "[2/5] Preparing FOMC data (download + features)"
echo "================================================================"

$PYTHON helpers/run_extract_fomc_features.py \
  --download-dir "$FOMC_DOWNLOAD_DIR" \
  --convert-dir  "$FOMC_CONVERT_DIR" \
  --features-dir "$FOMC_FEATURE_DIR" \
  --engine both

echo "[INFO] FOMC features expected under: $FOMC_FEATURE_DIR"
echo "       (features_librosa.parquet, features_opensmile.parquet, features_merged.parquet)"

###############################################################################
# 3) TRAIN MODEL – LIBROSA FEATURES ONLY
###############################################################################

echo
echo "================================================================"
echo "[3/5] Training model on LIBROSA features only"
echo "================================================================"

mkdir -p "$OUTPUT_LIBROSA"

$PYTHON helpers/run_audio_model.py \
  --infile "$TRAIN_FEATURE_LIBROSA" \
  --output-dir "$OUTPUT_LIBROSA" \
  --model-type baseline

echo "[INFO] Librosa model + predictions stored in: $OUTPUT_LIBROSA"

###############################################################################
# 4) TRAIN MODEL – OPENSMILE FEATURES ONLY
###############################################################################

echo
echo "================================================================"
echo "[4/5] Training model on OPENSMILE features only"
echo "================================================================"

mkdir -p "$OUTPUT_OPENSMILE"

$PYTHON helpers/run_audio_model.py \
  --infile "$TRAIN_FEATURE_OPENSMILE" \
  --output-dir "$OUTPUT_OPENSMILE" \
  --model-type baseline

echo "[INFO] OpenSMILE model + predictions stored in: $OUTPUT_OPENSMILE"

###############################################################################
# 5) TRAIN MODEL – COMBINED (LIBROSA + OPENSMILE) FEATURES
###############################################################################

echo
echo "================================================================"
echo "[5/5] Training model on COMBINED (Librosa + OpenSMILE) features"
echo "================================================================"

mkdir -p "$OUTPUT_COMBINED"

$PYTHON helpers/run_audio_model.py \
  --infile "$TRAIN_FEATURE_BOTH" \
  --output-dir "$OUTPUT_COMBINED" \
  --model-type baseline

echo "[INFO] Combined model + predictions stored in: $OUTPUT_COMBINED"

echo
echo "================================================================"
echo "[DONE] Full pipeline completed."
echo "================================================================"