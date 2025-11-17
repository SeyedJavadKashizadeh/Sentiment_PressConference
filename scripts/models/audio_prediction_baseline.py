"""
audio_prediction_baseline.py
====================

Inference script for applying a previously trained speech–emotion classifier
(see: `train_emotion_classifier.py`) to a CSV file of pre-extracted acoustic
features.

Given:
    • a trained Keras model (.h5 or SavedModel format),
    • a CSV containing feature rows (one sample per row),
    • an output path for predictions,

this script:

1) Loads the feature table and drops non-feature columns:
       - removes any column containing: "item", "contrast", "tonnetz",
         "path", "dataset", "emotion".
       - optional feature exclusions (mfcc / chroma / mel) can be enabled inside
         `processing_data`.

2) Ensures the resulting feature matrix has the correct shape for inference:
       (n_samples, n_features).

3) Loads the trained Keras model from `model_path`.

4) Computes predicted emotion indices (argmax of softmax outputs).

5) Maps predicted indices back to emotion labels
       ['happy', 'pleasant_surprise', 'neutral', 'sad', 'angry'].

6) Writes a CSV containing:
       item, emotion
   where `item` corresponds to the original input row index,
   and `emotion` is the predicted string label.

Typical Usage
-------------

    python predict_emotions.py \
        --model emotion_model.h5 \
        --infile input_features.csv \
        --outfile predictions.csv

Expected Input CSV Format
-------------------------
Must contain the acoustic feature columns used during training. Any additional
metadata columns containing "item", "path", "dataset", "emotion", "contrast",
"tonnetz" will be removed automatically.

Requirements
------------
- TensorFlow/Keras (legacy tf.keras compatibility flag set automatically)
- pandas, numpy
- The trained model must be compatible with the feature vector dimensions.
"""

import os

# Ensure compatibility with legacy tf.keras behavior if required
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
from typing import List

import tensorflow as tf
from tensorflow.keras.models import load_model, Model


EMOTIONS = ['happy', 'pleasant_surprise', 'neutral', 'sad', 'angry']

def processing_data(df: pd.DataFrame) -> np.ndarray:
    """
    Clean and prepare a feature DataFrame for emotion prediction.

    This function:
        - drops unsupported or metadata columns
        - optionally allows dropping specific feature groups (via commented lines)
        - converts the DataFrame to a numpy array
        - ensures output has shape (n_samples, n_features)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame of raw features.

    Returns
    -------
    np.ndarray
        Cleaned feature matrix of shape (n_samples, n_features).
    """
    # Drop metadata/non-feature columns
    df = df.drop([col for col in df.columns if "item" in col], axis=1)

    # Optional feature exclusions (uncomment if needed)
    # df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
    # df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
    # df = df.drop([col for col in df.columns if "mel" in col], axis=1)

    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion" in col], axis=1)

    x_pred = df.to_numpy()

    # Make sure model input is 2D
    if x_pred.ndim == 1:  # single row case
        x_pred = x_pred.reshape(1, -1)

    return x_pred


def audio_pred(model_path: str, infile: str, outfile: str, emotions:List[str] = EMOTIONS) -> None:
    """
    Predict emotions for an input feature CSV using a pretrained Keras model.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (.h5 or SavedModel directory).
    infile : str
        Path to a CSV file containing feature rows (one row per audio sample).
    outfile : str
        Path where the prediction CSV will be written.
    emotions : List[str]
        Emotions to be considered

    Output CSV Format
    -----------------
    Columns:
        item    – original row index from the input CSV
        emotion – predicted emotion label (string)

    Returns
    -------
    None
    """
    # Load input features
    df = pd.read_csv(infile, sep=",")

    # Supported emotion order must match training label order
    dictionary = {i: e for i, e in enumerate(emotions)}

    # Clean and vectorize features
    x_pred = processing_data(df)

    # Load trained Keras model
    model: Model = load_model(model_path)

    # Predict class indices
    y_pred = np.argmax(model.predict(x_pred), axis=-1)

    # Create output table
    output_data = pd.DataFrame(
        {
            "item": df.index,
            "emotion": pd.Series(y_pred).map(dictionary),
        }
    )

    # Save predictions
    output_data.to_csv(outfile, sep=",", index=False)
