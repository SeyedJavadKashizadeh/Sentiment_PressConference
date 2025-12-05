"""
Model factory for simple fully-connected emotion classifiers.

This module centralizes the creation of small feedforward neural network
classifiers used for emotion recognition. It exposes:

    • EMOTIONS_BASELINE / EMOTIONS_ADVANCED
        Canonical emotion label sets used by different experiments.

    • BASELINE_CONFIG / ADVANCED_DEFAULT_CONFIG
        Ready-to-use hyperparameter presets for `create_model(...)`.

    • create_model(...)
        Factory function that builds and compiles a Keras `Sequential` model
        given an input length, number of target classes, and a configuration
        dictionary (or equivalent keyword arguments).

Typical usage
-------------
The factory is usually used from training scripts that parse CLI arguments
and then construct a model as follows:

>>> from scripts.models.model_factory import (
...     create_model,
...     BASELINE_CONFIG,
...     EMOTIONS_BASELINE,
... )
...
>>> model = create_model(
...     input_length=128,
...     target_class=len(EMOTIONS_BASELINE),
...     **BASELINE_CONFIG,
... )

Assumptions
-----------
- Inputs are flat, fixed-length feature vectors (e.g. MFCCs, eGeMAPS, etc.),
  i.e. shape `(batch_size, input_length)`.
- The output is a categorical distribution over `target_class` emotions
  with a `softmax` activation and a sparse categorical cross-entropy loss.
- Optimizer and activation names are provided using small string aliases
  (e.g. `"adam"`, `"rmsprop"`, `"relu"`, `"gelu"`), which are normalized
  in `_get_activation` and `_get_optimizer`.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.activations import gelu
from typing import Callable


###############################################################################
# Emotion label sets
###############################################################################

EMOTIONS_BASELINE = ["happy", "pleasant_surprise", "neutral", "sad", "angry"]
EMOTIONS_ADVANCED = ["happy", "neutral", "sad", "angry"]


###############################################################################
# Configuration presets
###############################################################################

BASELINE_CONFIG = dict(
    num_layers=3,
    dense_units=200,
    dropout=0.3,
    optimizer="adam",
    learning_rate=1e-3,
    use_batchnorm=False,
    activation="relu",
    standardize_inputs=False,
)

ADVANCED_DEFAULT_CONFIG = dict(
    num_layers=3,
    dense_units=256,
    dropout=0.3,
    optimizer="adam",
    learning_rate=1e-3,
    use_batchnorm=True,
    activation="gelu",
    standardize_inputs=True,
)


###############################################################################
# Internal helpers
###############################################################################

def _get_activation(name: str) -> Callable:
    """
    Map a string identifier to a Keras activation function.

    Parameters
    ----------
    name : str
        Name of the activation function. Currently supported:
        - "relu"
        - "gelu"

    Returns
    -------
    Callable
        A callable activation function compatible with Keras layers.

    Raises
    ------
    ValueError
        If an unknown activation name is provided.
    """
    name = name.lower()
    if name == "relu":
        return tf.keras.activations.relu
    elif name == "gelu":
        # Use the imported GELU implementation to keep a single reference.
        return gelu
    else:
        raise ValueError(f"Unknown activation: {name!r}")


def _get_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """
    Map a string identifier to a compiled Keras optimizer instance.

    Parameters
    ----------
    name : str
        Optimizer name. Supported identifiers:
        - "adam"
        - "rmsprop"
        - "sgd"
    lr : float
        Learning rate to use for the optimizer.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Instantiated Keras optimizer with a gradient clipping norm.

    Notes
    -----
    - All optimizers apply `clipnorm=10.0` to help stabilize training
      when gradients become large.
    - For SGD, momentum and Nesterov acceleration are enabled.
    """
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=10.0)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.9,
            nesterov=True,
            clipnorm=10.0,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


###############################################################################
# Model factory
###############################################################################

def create_model(
    input_length: int,
    target_class: int,
    *,
    num_layers: int,
    dense_units: int,
    dropout: float,
    optimizer: str,
    learning_rate: float,
    use_batchnorm: bool,
    activation: str,
    ridge_penalty: float = 0.0,
    lasso_penalty: float = 0.0
) -> tf.keras.Model:
    """
    Build and compile a fully connected emotion classification model.

    The model is a simple `Sequential` stack of:
        Input -> [Dense (+BatchNorm) + Dropout] x num_layers -> Dense(softmax)

    Parameters
    ----------
    input_length : int
        Dimensionality of the input feature vector (per frame/sample).
    target_class : int
        Number of output emotion classes; used as the size of the final
        `Dense(..., activation="softmax")` layer.
    num_layers : int
        Total number of hidden Dense blocks. Each block contains:
        Dense(dense_units, activation), optional BatchNormalization,
        and Dropout(dropout).
    dense_units : int
        Number of units in each hidden Dense layer.
    dropout : float
        Dropout rate applied after each hidden block (value in [0, 1)).
    optimizer : str
        Optimizer identifier; passed to `_get_optimizer`, typically one of:
        "adam", "rmsprop", "sgd".
    learning_rate : float
        Learning rate for the selected optimizer.
    use_batchnorm : bool
        If True, a `BatchNormalization` layer is inserted after each Dense
        layer before dropout.
    activation : str
        Activation identifier passed to `_get_activation`, e.g. "relu" or "gelu".
    ridge_penalty:float
        Ridge penalty to include between the layers
    lasso_penalty:float
        Lasso penalty to include between the layers

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training.

    Examples
    --------
    >>> from scripts.models.model_factory import (
    ...     create_model,
    ...     BASELINE_CONFIG,
    ...     EMOTIONS_BASELINE,
    ... )
    ...
    >>> cfg = BASELINE_CONFIG.copy()
    >>> model = create_model(
    ...     input_length=128,
    ...     target_class=len(EMOTIONS_BASELINE),
    ...     **cfg,
    ... )
    >>> model.summary()
    """
    # Resolve activation function and optimizer from string identifiers.
    act_fn = _get_activation(activation)
    opt = _get_optimizer(optimizer, learning_rate)
    lambda_mix = tf.keras.regularizers.l1_l2(l1=lasso_penalty, l2=ridge_penalty)

    # -------------------------------------------------------------------------
    # Model topology definition
    # -------------------------------------------------------------------------
    model = Sequential(name="emotion_classifier")


    # Explicit input layer (preferred over input_dim / input_shape on Dense).
    model.add(Input(shape=(input_length,), name="input"))
    
    # First dense block
    model.add(Dense(dense_units, activation=act_fn, name="dense_1", kernel_regularizer = lambda_mix))
    if use_batchnorm:
        model.add(BatchNormalization(name="bn_1"))
    model.add(Dropout(dropout, name="dropout_1"))

    for i in range(2, num_layers + 1):
        model.add(Dense(dense_units, activation=act_fn, name=f"dense_{i}", kernel_regularizer = lambda_mix))
        if use_batchnorm:
            model.add(BatchNormalization(name=f"bn_{i}"))
        model.add(Dropout(dropout, name=f"dropout_{i}"))

    # Output layer: probability distribution over emotions
    model.add(
        Dense(
            target_class,
            activation="softmax",
            name="output",
        )
    )

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    return model
