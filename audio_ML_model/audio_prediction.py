import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model  # typed import


def processing_data(df):
    df = df.drop([col for col in df.columns if "item" in col], axis=1)
    # Uncomment to drop a feature
##    df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
##    df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
##    df = df.drop([col for col in df.columns if "mel" in col], axis=1)
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion" in col], axis=1)
    x_pred = df.to_numpy()
    # Ensure 2D input (n_samples, n_features)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)
    return x_pred

def audio_pred(model_path, infile, outfile):
    df=pd.read_csv(infile, sep=',')
    emotions = ['happy', 'pleasant_surprise', 'neutral', 'sad', 'angry']
    dictionary = {0: 'happy', 1: 'pleasant_surprise', 2: 'neutral', 3: 'sad', 4: 'angry'}
    x_pred = processing_data(df)

    model: Model = load_model(model_path)
    y_pred = np.argmax(model.predict(x_pred), axis=-1)
    output_data = pd.DataFrame({
        'item': df.index,
        'emotion': pd.Series(y_pred).map(dictionary)
    })
    output_data.to_csv(outfile, sep=',', index=False)
