import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import pandas as pd
import numpy as np
import tensorflow as tf
from tf.keras.models import load_model, Model  # typed import


def processing_data(df):
    df = df.drop([col for col in df.columns if "item" in col], axis=1)
    # Uncomment to drop a feature
##    df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
##    df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
##    df = df.drop([col for col in df.columns if "mel" in col], axis=1)
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)

    x_pred = df.to_numpy()
    return x_pred

infile = 'PATH TO THE CSV FILE CONTAINING EXTRACTED FEATURES OF PREDICTION SAMPLE'
outfile = 'PATH TO THE OUTPUT FILE'
emotions = ['happy', 'ps', 'neutral', 'sad', 'angry']
dictionary = {0: 'happy', 1: 'ps', 2: 'neutral', 3: 'sad', 4: 'angry'}
df=pd.read_csv(infile, sep='\t')
x_pred = processing_data(df)

      
model: Model = load_model('PATH TO THE SAVED MODEL')
y_pred = np.argmax(model.predict(x_pred), axis=-1)
output_data = pd.DataFrame({
    'item': df['item'],
    'emotion': pd.Series(y_pred).map(dictionary)
})
output_data['item'] = df['item']
output_data.to_csv(outfile, sep='\t', index=False)
