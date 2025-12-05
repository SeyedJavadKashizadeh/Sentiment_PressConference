# scripts/models/utils_models.py

"""
Utility functions for preparing data for emotion classification models.

This module currently provides tools to:
    - Load feature CSV files exported from the feature extraction pipeline.
    - Filter, clean, and reshape those features.
    - Produce balanced train/test splits for a predefined set of emotions.

The main entry point is `split_data`, which:
    - Reads a CSV file with per-utterance features and labels.
    - Drops non-feature / metadata columns (e.g. paths, dataset names).
    - Selects only samples whose emotion label is in a given list.
    - Constructs a balanced train/test split by:
        * identifying the minimum sample count across the target emotions,
        * sampling a fixed proportion of that minimum for training,
        * assigning remaining samples from each class to the test set.
    - Returns NumPy arrays for features and labels, along with mappings
      between integer indices and emotion names.

Typical usage
-------------
>>> from scripts.models.utils_models import split_data
>>>
>>> emotions = ["happy", "neutral", "sad", "angry"]
>>> (
...     int2emotions,
...     emotions2int,
...     x_train,
...     y_train,
...     x_test,
...     y_test,
... ) = split_data(
...     emotions=emotions,
...     filename="features.csv",
...     train_set=0.8,
...     seed=42,
... )

Assumptions
-----------
- The input CSV has at least the following columns:
    * "emotion_name" (categorical label as a string)
    * various numeric feature columns
    * optional metadata columns like "contrast*", "tonnetz*", "path",
      "dataset", "emotion_code", which will be dropped.
- Labels are stored as strings in "emotion_name" and are filtered to the
  subset specified in `emotions`.
- A simple per-class sampling strategy is sufficient for balancing.
"""

from __future__ import annotations

###############################################################################
# Standard libraries
###############################################################################
import math
from typing import Dict, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from pathlib import Path


###############################################################################
# Data splitting utilities
###############################################################################
def split_data(
    emotions: Sequence[str],
    filename: Union[str, Path],
    train_set: float,
    seed: int = 42,
) -> Tuple[
    Dict[int, str],
    Dict[str, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load a feature CSV, filter columns, and create a balanced train/test split.

    The function:
        - loads the CSV from `filename`
        - drops non-feature columns (contrast, tonnetz, path, emotion_code)
          but KEEPS the 'dataset' column so we can evaluate per dataset later
        - filters to the requested `emotions` (when computing class counts and
          building the splits)
        - computes the minimum per-class count among `emotions` and uses it to:
            * sample `floor(train_set * min_count)` examples per emotion
              for training
            * assign the remaining examples of each class to testing
        - returns NumPy arrays for X_train, y_train, X_test, y_test, along with
          dataset labels for train/test, and the label mapping dictionaries.

    Parameters
    ----------
    emotions : Sequence[str]
        Emotion labels to keep, e.g. ["happy", "sad", "neutral", ...].
        The order of this sequence defines the integer label mapping.
    filename : str or pathlib.Path
        Path to the CSV file containing features and labels.
    train_set : float
        Proportion of the smallest class used for the training set
        (e.g. 0.8 for 80% train / 20% test). Must be in the interval (0, 1].
    seed : int, optional
        Random seed for the sampling procedure, by default 42.

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
    ds_train : np.ndarray
        Dataset labels (e.g. "EMODB", "RAVDESS", "TESS") for the training set,
        shape (n_train,).
    ds_test : np.ndarray
        Dataset labels for the test set, shape (n_test,).

    Raises
    ------
    ValueError
        If `train_set` is not in the interval (0, 1].
    """
    if not (0 < train_set <= 1.0):
        raise ValueError(f"`train_set` must be in (0, 1], got {train_set!r}.")

    # -------------------------------------------------------------------------
    # Load CSV and drop non-feature / metadata columns
    # -------------------------------------------------------------------------
    df = pd.read_csv(filename, sep=",")

    # Optionally drop feature groups if desired:
    # df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
    # df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
    # df = df.drop([col for col in df.columns if "mel" in col], axis=1)

    # Drop unwanted feature groups / metadata by default
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    # KEEP 'dataset' so we can evaluate per corpus later
    df = df.drop([col for col in df.columns if "emotion_code" in col], axis=1)

    # -------------------------------------------------------------------------
    # Compute class counts restricted to the requested emotions
    # -------------------------------------------------------------------------
    counts = [
        len(df[df.emotion_name == emotion])
        for emotion in df.emotion_name.unique()
        if emotion in emotions
    ]

    int2emotions: Dict[int, str] = {i: e for i, e in enumerate(emotions)}
    emotions2int: Dict[str, int] = {v: k for k, v in int2emotions.items()}

    min_count = math.floor(min(counts) * train_set)

    # Initialize containers for balanced train/test splits
    x_train, x_test = pd.DataFrame(), pd.DataFrame()
    y_train = pd.DataFrame(columns=["emotion_name"])
    y_test = pd.DataFrame(columns=["emotion_name"])
    ds_train = pd.DataFrame(columns=["dataset"])
    ds_test = pd.DataFrame(columns=["dataset"])

    # -------------------------------------------------------------------------
    # Per-emotion sampling
    # -------------------------------------------------------------------------
    for emotion in emotions:
        temp = df.loc[df.emotion_name == emotion]

        train_temp = temp.sample(n=min_count, random_state=seed)

        test_temp = (
            pd.merge(temp, train_temp, how="outer", indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )

        # feature columns = all except labels and dataset
        feature_cols = [c for c in df.columns if c not in ("emotion_name", "dataset")]

        x_train = pd.concat([x_train, train_temp[feature_cols]], axis=0)
        y_train = pd.concat(
            [y_train, pd.DataFrame(train_temp["emotion_name"])], axis=0
        )
        ds_train = pd.concat(
            [ds_train, pd.DataFrame(train_temp["dataset"])], axis=0
        )

        x_test = pd.concat([x_test, test_temp[feature_cols]], axis=0)
        y_test = pd.concat(
            [y_test, pd.DataFrame(test_temp["emotion_name"])], axis=0
        )
        ds_test = pd.concat(
            [ds_test, pd.DataFrame(test_temp["dataset"])], axis=0
        )

    # -------------------------------------------------------------------------
    # Diagnostics and conversion to NumPy arrays
    # -------------------------------------------------------------------------
    print(
        "Training features:{}; Training output:{}; "
        "Testing features:{}; Testing output:{}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()
    x_test_np = x_test.to_numpy()
    y_test_np = y_test.to_numpy()
    ds_train_np = ds_train["dataset"].to_numpy()
    ds_test_np = ds_test["dataset"].to_numpy()

    return (
        int2emotions,
        emotions2int,
        x_train_np,
        y_train_np,
        x_test_np,
        y_test_np,
        ds_train_np,
        ds_test_np,
    )
