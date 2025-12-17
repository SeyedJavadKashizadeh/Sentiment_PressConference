# Sentiment_PressConference

Speech Emotion Recognition (SER) pipeline for analyzing the emotional tone of Federal Reserve (FOMC) press conferences.

This project trains emotion classifiers on standard datasets (RAVDESS, TESS, EmoDB), predicts emotions on FOMC audio data, and evaluates the results against the "Voice Tone" metric defined in the *Voice of Monetary Policy (VOMP)* paper.

## Features

- **Multi-Dataset Training**: Downloads and harmonizes RAVDESS, TESS, and EmoDB into a unified format.
- **Feature Extraction**:
  - **Librosa**: 193-dimensional vector (MFCCs, Chroma, Mel, Contrast, Tonnetz).
  - **OpenSMILE**: eGeMAPSv02 Functionals.
- **Modeling**:
  - **Baseline**: Simple MLP (Paper replication).
  - **Advanced**: Optimized MLP with Batch Normalization, GELU activation, and regularization.
- **Evaluation**: Replicates the VOMP "Voice Tone" metric and compares predictions against reference data.
- **Visualization**: Extensive EDA and hyperparameter search visualization tools.

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg**: Required for audio conversion.
   - Windows: Download and add to PATH (or specify path in CLI).
   - Linux: `sudo apt-get install ffmpeg`
3. **Kaggle Account**: For downloading training datasets.
   - Ensure `~/.kaggle/kaggle.json` exists or environment variables are set.
4. **Hugging Face Account**: For downloading the FOMC dataset.
   - You need access to `FedSentimentLab/Fed_audio_text_video`.

### Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r package_requirements.txt
   ```
3. Create a `.env` file in the root directory for your Hugging Face token:
   ```env
   HF_TOKEN=hf_your_token_here
   ```

## Quick Start

To run the entire pipeline (download -> extract -> train -> predict -> eval), use the provided shell script:

```bash
bash run_pipeline.sh
```

## Usage Guide (Step-by-Step)

The project uses a central dispatcher `run.py` for all tasks.

### 1. Prepare Training Data

Download datasets, convert audio to 16kHz mono, and extract features.

```bash
python run.py extract-training-features \
    --gender female \
    --emotions happy neutral sad angry \
    --engine both \
    --out-file data_training/merged_with_features.csv
```

*   **--gender**: Filter by speaker gender (`male`, `female`).
*   **--emotions**: List of emotions to keep (IDs or names).
*   **--engine**: `librosa`, `opensmile`, or `both`.

**Emotion Mapping:**
The pipeline normalizes emotions to a universal ID space. **Note:** "Calm" is collapsed into "Neutral".
- 1: Neutral (includes Calm)
- 3: Happy
- 4: Sad
- 5: Angry
- 6: Fear
- 7: Disgust
- 8: Pleasant Surprise
- 9: Boredom

### 2. Prepare FOMC Data

Download FOMC audio from Hugging Face and extract features.

```bash
python run.py extract-fomc-features \
    --engine both \
    --download-dir data_fomc/raw_data \
    --features-dir data_fomc/features
```

### 3. Exploratory Data Analysis (EDA)

Generate plots for feature distributions and t-SNE embeddings.

```bash
python run.py eda-analysis \
    --librosa-csv data_training/merged_with_features_librosa.csv \
    --opensmile-csv data_training/merged_with_features_opensmile.csv \
    --save-dir outputs/eda_analysis
```

### 4. Model Training

#### Baseline Model
Trains a simple MLP (Linear activation, Dropout) on Librosa features.

```bash
python run.py train-model \
    --mode baseline \
    --infile data_training/merged_with_features_librosa.csv \
    --model-path outputs/baseline/model.keras
```

#### Advanced Model
Trains a deeper network with Batch Normalization, GELU, and standardization.

```bash
python run.py train-model \
    --mode advanced \
    --infile data_training/merged_with_features.csv \
    --model-path outputs/advanced/model.keras \
    --standardize-inputs \
    --use-batchnorm \
    --activation gelu
```

### 5. Cross-Validation & Hyperparameter Search

Run grid search for the advanced model configuration.

```bash
python run.py cross-validation \
    --training-dataset data_training/merged_with_features.csv \
    --cv-splits 5
```

Visualize results:
```bash
python run.py visualize-experiments \
    --results-csv outputs/experiments/results_FINAL.csv \
    all
```

### 6. Prediction

Predict emotions on the FOMC dataset.

```bash
python run.py predict-model \
    --mode advanced \
    --weights outputs/advanced/model.keras \
    --infile data_fomc/features/features_merged.parquet \
    --outfile outputs/advanced/predictions_fomc.csv
```

### 7. VOMP Evaluation

Compare predictions against the VOMP reference metrics.

```bash
python run.py vomp-eval \
    --vomp-dir vomp_data \
    --model-csvs outputs/advanced/predictions_fomc.csv \
    --out-dir outputs/vomp_eval \
    --plot
```

**Metric:**
`Voice Tone = (Positive - Negative) / (Positive + Negative)`
*   Positive: Happy, Pleasant Surprise
*   Negative: Sad, Angry

## Project Structure

- `helpers/`: CLI entry points for each pipeline stage.
- `scripts/`: Core logic modules.
  - `download_*`: Dataset downloaders.
  - `extract_*`: Feature extraction logic.
  - `models/`: Model definitions and training utilities.
  - `visualization/`: Plotting and evaluation logic.
- `data_training/`: Stores training raw audio and features.
- `data_fomc/`: Stores FOMC raw audio and features.
- `outputs/`: Stores trained models, predictions, and plots.

## Reproducibility

The pipeline uses `utils.set_global_seed()` to set seeds for Python, NumPy, and TensorFlow to ensure reproducible results.