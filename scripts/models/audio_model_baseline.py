"""
audio_model_baseline.py
===========================

Feed-forward neural network training script for speech emotion recognition on
pre-extracted acoustic features (e.g. MFCCs, chroma, mel, contrast, tonnetz).

Given a CSV of features (one row per utterance), this script:

1) Loads the feature table and keeps only the desired feature subsets:
       - drops contrast / tonnetz / path / dataset / emotion_code columns
         (can be changed in `split_data` if needed).

2) Builds a *balanced* train/test split across a fixed set of emotions by:
       - computing per-class counts
       - sampling the same number of examples per emotion for training
       - using the remaining examples of each class for testing.

3) One-hot encodes emotion labels and trains a dense neural network:
       - 3 hidden Dense layers with dropout
       - output layer with softmax over emotion classes
       - loss = categorical_crossentropy, optimizer = Adam.

4) Evaluates the model on the held-out test set:
       - overall accuracy
       - confusion matrix as a pandas.DataFrame.

The main entry point is `train_model(model_path, infile)`, which returns:
    - the trained Keras model
    - int2emotions: mapping from integer index to emotion string
    - emotions2int: mapping from emotion string to integer index

Expected CSV format
-------------------
Minimal columns required (others will be dropped in `split_data`):
    - emotion_name : string label (e.g. "happy", "sad", "neutral")
    - feature columns: any subset of mfcc/chroma/mel/etc. (numeric)

Columns explicitly dropped:
    - any containing: "contrast", "tonnetz", "path", "dataset", "emotion_code"

Notes
-----
- The TF/Keras legacy flag `TF_USE_LEGACY_KERAS=1` is set to ensure compatibility
  in environments where the old `tf.keras` behavior is required.
- To change the list of emotions, modify the `emotions` list in `train_model`.
- To switch feature subsets (e.g. drop MFCCs or chroma), uncomment lines inside
  `split_data` where indicated.
"""

import os
from pathlib import Path

# Ensure TensorFlow uses legacy tf.keras behavior (if needed in your environment).
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import pandas as pd
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.activations import gelu
from tensorflow.keras.optimizers import (
    SGD,
    RMSprop,
    Adam,
    Adadelta,
    Adagrad,
    Adamax,
    Nadam,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    History,
    ReduceLROnPlateau,
    CSVLogger,
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold

EMOTIONS = ['happy', 'neutral', 'sad', 'angry']

def split_data(emotions, filename, train_set):
    """
    Load a feature CSV, filter columns, and create a balanced train/test split.

    The function:
        - loads the CSV from `filename`
        - drops non-feature columns (contrast, tonnetz, path, dataset, emotion_code)
        - filters to the requested `emotions`
        - computes the minimum per-class count and uses it to:
              * sample `train_set * min_count` examples per emotion for training
              * assign the rest of each class to testing
        - returns numpy arrays for X_train, y_train, X_test, y_test and
          the label mapping dictionaries.

    Parameters
    ----------
    emotions : list of str
        Emotion labels to keep (e.g. ['happy', 'sad', 'neutral', ...]).
    filename : str or Path
        Path to the CSV file containing features and labels.
    train_set : float
        Proportion of the smallest class used for the training set
        (e.g. 0.8 for 80% train / 20% test).

    Returns
    -------
    int2emotions : dict[int, str]
        Mapping from integer index to emotion string.
    emotions2int : dict[str, int]
        Mapping from emotion string to integer index.
    x_train : np.ndarray
        Training features of shape (n_train, n_features).
    y_train : np.ndarray
        Training labels as strings, shape (n_train, 1).
    x_test : np.ndarray
        Testing features of shape (n_test, n_features).
    y_test : np.ndarray
        Testing labels as strings, shape (n_test, 1).
    """
    df = pd.read_csv(filename, sep=",")

    # Uncomment to drop specific feature groups if desired:
    # df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
    # df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
    # df = df.drop([col for col in df.columns if "mel" in col], axis=1)

    # Drop unwanted feature groups / metadata by default
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion_code" in col], axis=1)

    # Balanced training sample construction
    y_df = df["emotion_name"]
    counts = [
        len(df[df.emotion_name == emotion])
        for emotion in df.emotion_name.unique()
        if emotion in emotions
    ]

    int2emotions = {i: e for i, e in enumerate(emotions)}
    emotions2int = {v: k for k, v in int2emotions.items()}

    # Number of training examples per emotion
    min_count = math.floor(min(counts) * train_set)

    x_train, x_test = pd.DataFrame(), pd.DataFrame()
    y_train = pd.DataFrame(columns=["emotion_name"])
    y_test = pd.DataFrame(columns=["emotion_name"])

    for emotion in emotions:
        # All samples of this emotion
        temp = df.loc[df.emotion_name == emotion]

        # Randomly sample training subset
        train_temp = temp.sample(n=min_count, random_state=100)

        # Remaining samples become testing data:
        # left is 'temp', right is 'train_temp', keep only rows not in train_temp
        test_temp = (
            pd.merge(temp, train_temp, how="outer", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )

        # Append to global train/test sets
        x_train = pd.concat([x_train, train_temp.drop(columns=["emotion_name"])])
        y_train = pd.concat([y_train, pd.DataFrame(train_temp["emotion_name"])])
        x_test = pd.concat([x_test, test_temp.drop(columns=["emotion_name"])])
        y_test = pd.concat([y_test, pd.DataFrame(test_temp["emotion_name"])])

    print(
        "Training features:{}; Training output:{}; Testing features:{}; Testing output:{}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    return int2emotions, emotions2int, x_train, y_train, x_test, y_test


def test_score(model, y_test, x_test):
    """Compute accuracy score assuming `y_test` is one-hot encoded."""
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    return accuracy_score(y_true, y_pred)


def conf_matrix(model, y_test, x_test, emotions2int, emotions):
    """
    Compute confusion matrix as a pandas.DataFrame for a trained model.

    Assumes `y_test` is one-hot encoded and `emotions2int` provides the mapping
    from emotion names to integer indices.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model used for prediction.
    y_test : np.ndarray
        One-hot encoded true labels.
    x_test : np.ndarray
        Test features.
    emotions2int : dict[str, int]
        Mapping from emotion name to integer index.
    emotions : list[str]
        Ordered list of emotions to fix the confusion matrix label order.

    Returns
    -------
    pd.DataFrame
        Confusion matrix with rows indexed as t_<emotion>,
        columns as p_<emotion>.
    """
    # Convert one-hot encodings to integer labels
    y_true = [np.argmax(v) for v in y_test]

    # Predict class probabilities and take argmax
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=-1)

    cm = confusion_matrix(
        y_true, y_pred, labels=[emotions2int[e] for e in emotions]
    )
    return pd.DataFrame(
        cm,
        index=[f"t_{e}" for e in emotions],
        columns=[f"p_{e}" for e in emotions],
    )


def train_model(model_path, infile, metrics_dir, emotions:List = EMOTIONS):
    """
    Train a dense neural network for emotion classification on pre-extracted features.

    This function:
        - defines the set of target emotions,
        - loads and balances the dataset via `split_data`,
        - one-hot encodes labels,
        - builds and trains a feed-forward network,
        - evaluates accuracy and prints the confusion matrix,
        - saves computed metrics to `metrics_dir`,
        - saves the best model to `model_path` (via ModelCheckpoint).

    Parameters
    ----------
    model_path : str or Path
        Destination path where the best Keras model will be saved.
    infile : str or Path
        Path to the CSV file containing features and labels.
    metrics_dir : str or Path
        CSV where the evaluation metrics will be stored.
    emotions : List[str]
        List of emotions to be analyzed

    Returns
    -------
    model : tf.keras.Model
    int2emotions : dict[int, str]
    emotions2int : dict[str, int]
    """

    # Build balanced train/test split
    int2emotions, emotions2int, x_train, y_train, x_test, y_test = split_data(
        emotions, infile, train_set=0.8
    )  

    # One-hot encode labels
    y_train = to_categorical([emotions2int[str(e[0])] for e in y_train])
    y_test = to_categorical([emotions2int[str(e[0])] for e in y_test])
    
    target_class = len(emotions)
    input_length = x_train.shape[1]

    dense_units = 200
    dropout = 0.3
    loss = "categorical_crossentropy"
    optimizer = "adam"

    model = Sequential()
    model.add(Dense(dense_units, input_dim=input_length))
    model.add(BatchNormalization())
    model.add(Activation(gelu)) 
    model.add(Dropout(dropout))

    model.add(Dense(dense_units))
    model.add(BatchNormalization())
    model.add(Activation(gelu)) 
    model.add(Dropout(dropout))

    model.add(Dense(dense_units))
    model.add(BatchNormalization())
    model.add(Activation(gelu)) 
    model.add(Dropout(dropout))

    model.add(Dense(target_class, activation="softmax"))
    
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[CategoricalAccuracy(), Precision(), Recall()],
    )

    # Callbacks
    checkpointer = ModelCheckpoint(
        model_path, save_best_only=True, monitor="val_loss"
    )
    lr_reduce = ReduceLROnPlateau(
        monitor="val_loss", factor=0.9, patience=20, min_lr=0.000001
    )

    # Train
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=1000,
        validation_data=(x_test, y_test),
        callbacks=[checkpointer, lr_reduce],
    )

    '''
        Checking accuracy score and confusion matrix
    '''
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    y_test = [np.argmax(i, out=None, axis=None) for i in y_test]

    print(accuracy_score(y_true=y_test, y_pred=y_pred))

    matrix = confusion_matrix(y_test, y_pred,
                              labels=[emotions2int[e] for e in emotions])
    matrix = pd.DataFrame(matrix, index=[f"t_{e}" for e in emotions],columns=[f"p_{e}" for e in emotions])
    print(matrix)
    print(classification_report(y_test, y_pred, target_names=emotions, digits=4))
    
    return model, int2emotions, emotions2int

def train_model_kfold(model_path, infile,n_splits=5, emotions:List = EMOTIONS):

    df = pd.read_csv(infile, sep=",")

    # Uncomment to drop specific feature groups if desired:
    # df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
    # df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
    # df = df.drop([col for col in df.columns if "mel" in col], axis=1)

    # Drop unwanted feature groups / metadata by default
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion_code" in col], axis=1)

    # Balanced training sample construction
    X = df.drop(columns=["emotion_name"]).to_numpy()
    y = df["emotion_name"].to_numpy()

    mask = np.array([label in emotions for label in y])
    X = X[mask]
    y = y[mask] 

    int2emotions = {i: e for i, e in enumerate(emotions)}
    emotions2int = {v: k for k, v in int2emotions.items()}

    y_int = np.array([emotions2int[label] for label in y])
    
    target_class = len(emotions)
    input_length = X.shape[1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []
    fold = 1

    for train_idx, test_idx in skf.split(X, y_int):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y_int[train_idx], y_int[test_idx]

        # One-hot encode
        y_train = to_categorical(y_train_raw, num_classes=target_class)
        y_test = to_categorical(y_test_raw, num_classes=target_class)

        dense_units = 200
        dropout = 0.3
        loss = "categorical_crossentropy"
        optimizer = "adam"

        model = Sequential()
        model.add(Dense(dense_units, input_dim=input_length))
        model.add(BatchNormalization())
        model.add(Activation(gelu)) 
        model.add(Dropout(dropout))

        model.add(Dense(dense_units))
        model.add(BatchNormalization())
        model.add(Activation(gelu)) 
        model.add(Dropout(dropout))

        model.add(Dense(dense_units))
        model.add(BatchNormalization())
        model.add(Activation(gelu)) 
        model.add(Dropout(dropout))

        model.add(Dense(target_class, activation="softmax"))

        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[CategoricalAccuracy(), Precision(), Recall()],
        )

        # Callbacks
        checkpointer = ModelCheckpoint(
            f"{model_path}_fold{fold}.h5",
            save_best_only=True,
            monitor="val_loss"
        )
        lr_reduce = ReduceLROnPlateau(
            monitor="val_loss", factor=0.9, patience=20, min_lr=1e-6
        )

        # Train
        model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=1000,
            validation_data=(X_test, y_test),
            callbacks=[checkpointer, lr_reduce],
            verbose=1
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        y_true = y_test_raw

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

        # Save fold metrics
        all_results.append({
            "fold": fold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        fold += 1

    df = pd.DataFrame(all_results)

    print("\ninal K-Fold Results")
    print(df)

    print("\nAverages:")
    print(df.mean())

    return df, int2emotions, emotions2int