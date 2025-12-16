"""
Plotting and evaluation helpers for emotion-classification experiments.

This module groups small, reusable utilities to visualize training curves and
summarize classification performance. It provides:

    • plot_history(...)
        Save loss and accuracy curves from a Keras/TF history DataFrame.

    • plot_confusion_matrix_heatmap(...)
        Render a confusion matrix (optionally row-normalized) as a heatmap.

    • plot_per_class_metrics_bars(...)
        Plot per-class precision / recall / F1 as grouped bars.

    • compute_per_dataset_metrics(...)
        Compute per-dataset macro precision/recall/F1 (useful for multi-corpus tests).

    • plot_per_dataset_f1_bar(...)
        Visualize dataset-level macro F1 as a bar plot.

Typical usage
-------------
Assuming you have:
- df_history: a DataFrame built from `history.history`
- cm: a confusion matrix (n_classes x n_classes)
- y_true / y_pred: integer label arrays aligned with your test set
- ds_test: dataset IDs aligned with your test set

>>> from pathlib import Path
>>> df_history = pd.DataFrame(history.history)
>>> plot_history(df_history, Path("loss.png"), Path("acc.png"))

>>> plot_confusion_matrix_heatmap(cm, emotions, Path("cm.png"), normalize=True)

>>> prec, rec, f1, _ = precision_recall_fscore_support(
...     y_true, y_pred, labels=[emotions2int[e] for e in emotions], average=None
... )
>>> plot_per_class_metrics_bars(emotions, prec, rec, f1, Path("per_class.png"))

>>> df_ds = compute_per_dataset_metrics(y_true, y_pred, ds_test, emotions, emotions2int)
>>> plot_per_dataset_f1_bar(df_ds, Path("per_dataset_f1.png"))

Assumptions
-----------
- `df_history` contains at least "loss" and (optionally) "val_loss".
  For accuracy, it contains one of ("accuracy", "acc") and corresponding
  validation keys ("val_accuracy" or "val_acc").
- Confusion matrices are shaped [n_classes, n_classes] with axes aligned to
  the provided `emotions` list.
- Label arrays `y_true` and `y_pred` contain integer class indices compatible
  with `emotions2int`.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support


###############################################################################
# Training curve plots
###############################################################################

def plot_history(df_history, loss_outfile, acc_outfile, use_log_scale=True):
    """
    Plot training history curves and save to disk.

    Parameters
    ----------
    df_history : pd.DataFrame
        DataFrame typically built from `history.history` (Keras/TF), containing
        per-epoch metrics such as "loss", "val_loss", "accuracy"/"acc",
        and "val_accuracy"/"val_acc".
    loss_outfile : str | Path
        Output path for the loss plot (PNG).
    acc_outfile : str | Path
        Output path for the accuracy plot (PNG).
    use_log_scale : bool, default=True
        If True, the loss plot uses a logarithmic y-axis to make early-epoch
        differences more visible.

    Returns
    -------
    None
        Side effect: writes PNG files to `loss_outfile` and `acc_outfile` if
        the required keys are present.
    """

    # -------------------------------------------------------------------------
    # Loss plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["loss"], label="train_loss", linewidth=2)

    # Validation loss is optional.
    if "val_loss" in df_history:
        plt.plot(df_history["val_loss"], label="val_loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    if use_log_scale:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(loss_outfile), dpi=150)
    plt.close()
    print(f"[saved loss plot] {loss_outfile}")

    # -------------------------------------------------------------------------
    # Accuracy plot
    # -------------------------------------------------------------------------
    # Keras uses either "accuracy" or "acc" depending on version/config.
    acc_key = None
    val_acc_key = None
    for k in ["accuracy", "acc"]:
        if k in df_history:
            acc_key = k
        if f"val_{k}" in df_history:
            val_acc_key = f"val_{k}"

    # If accuracy keys are missing, avoid crashing and just skip.
    if acc_key is None or val_acc_key is None:
        print("[warning] accuracy keys not found in history; skipping accuracy plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df_history[acc_key], label="train_accuracy", linewidth=2)
    plt.plot(df_history[val_acc_key], label="val_accuracy", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(acc_outfile), dpi=150)
    plt.close()
    print(f"[saved accuracy plot] {acc_outfile}")


###############################################################################
# Confusion matrix + per-class metric plots
###############################################################################

def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    emotions: list[str],
    outfile: Path,
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix as a heatmap and save it to disk.

    By default, the confusion matrix is row-normalized, so each row sums to 1
    and values can be read as "proportion of true class predicted as ...".

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix with shape [n_classes, n_classes].
        Rows correspond to true labels, columns to predicted labels.
    emotions : list[str]
        Class names, in the same order as the matrix axes.
    outfile : Path
        Output path for the heatmap PNG.
    normalize : bool, default=True
        If True, row-normalize the confusion matrix before plotting.

    Returns
    -------
    None
        Side effect: writes a PNG to `outfile`.
    """
    if normalize:
        # Convert to float and divide each row by its sum.
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)

        # Prevent division by zero for classes not present in y_true.
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150)
    plt.close()
    print(f"[saved confusion matrix] {outfile}")


def plot_per_class_metrics_bars(
    emotions: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    outfile: Path,
) -> None:
    """
    Plot per-class precision, recall, and F1 as grouped bars.

    Parameters
    ----------
    emotions : list[str]
        Class names, length = n_classes.
    precision : np.ndarray
        Per-class precision scores, length = n_classes.
    recall : np.ndarray
        Per-class recall scores, length = n_classes.
    f1 : np.ndarray
        Per-class F1 scores, length = n_classes.
    outfile : Path
        Output path for the bar plot PNG.

    Returns
    -------
    None
        Side effect: writes a PNG to `outfile`.
    """
    x = np.arange(len(emotions))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x,         recall,   width, label="Recall")
    plt.bar(x + width, f1,       width, label="F1")

    plt.xticks(x, emotions, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-class precision / recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150)
    plt.close()
    print(f"[saved per-class metrics] {outfile}")


###############################################################################
# Per-dataset evaluation (multi-corpus testing)
###############################################################################

def compute_per_dataset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ds_test: np.ndarray,
    emotions: list[str],
    emotions2int: dict[str, int],
) -> pd.DataFrame:
    """
    Compute macro precision/recall/F1 per dataset identifier.

    This is useful when your test set is a concatenation of multiple corpora
    (datasets) and you want a quick breakdown of performance per source.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels for each example.
    y_pred : np.ndarray
        Predicted integer labels for each example.
    ds_test : np.ndarray
        Dataset identifiers aligned with y_true/y_pred. Can be strings or ints.
    emotions : list[str]
        Class names to include in the metric computation.
    emotions2int : dict[str, int]
        Mapping from class name to integer label index.

    Returns
    -------
    pd.DataFrame
        One row per dataset with:
            - dataset
            - precision_macro
            - recall_macro
            - f1_macro
            - n_samples
    """
    label_indices = [emotions2int[e] for e in emotions]
    rows = []

    for ds in sorted(np.unique(ds_test)):
        mask = ds_test == ds
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        # Macro-average over the specified label set to keep comparisons stable.
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", labels=label_indices
        )

        rows.append(
            {
                "dataset": ds,
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "f1_macro": float(f1),
                "n_samples": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def plot_per_dataset_f1_bar(df_ds: pd.DataFrame, outfile: Path) -> None:
    """
    Plot dataset-level macro F1 as a bar chart.

    Parameters
    ----------
    df_ds : pd.DataFrame
        DataFrame returned by `compute_per_dataset_metrics`. Must contain:
        - dataset
        - f1_macro
    outfile : Path
        Output path for the bar plot PNG.

    Returns
    -------
    None
        Side effect: writes a PNG to `outfile`.
    """
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_ds, x="dataset", y="f1_macro")
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1")
    plt.title("Per-dataset macro F1")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[saved per-dataset F1 plot] {outfile}")
