import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

def plot_history(df_history, loss_outfile, acc_outfile, use_log_scale=True):
    """
    df_history  : pandas DataFrame from history.history
    loss_outfile: path for the loss PNG
    acc_outfile : path for the accuracy PNG
    use_log_scale: if True, use log scale on the loss y-axis
    """

    ### Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_history["loss"], label="train_loss", linewidth=2)
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

    ### Accuracy plot
    acc_key = None
    val_acc_key = None
    for k in ["accuracy", "acc"]:
        if k in df_history:
            acc_key = k
        if f"val_{k}" in df_history:
            val_acc_key = f"val_{k}"

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

def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    emotions: list[str],
    outfile: Path,
    normalize: bool = True,
) -> None:
    """
    Plot a (optionally row-normalized) confusion matrix as a heatmap.

    cm       : confusion matrix (shape [n_classes, n_classes])
    emotions : list of emotion labels in the same order as cm axes
    outfile  : path to save the PNG
    """
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
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

    emotions : list of emotion labels, length = n_classes
    precision, recall, f1 : arrays of length n_classes
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

def compute_per_dataset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ds_test: np.ndarray,
    emotions: list[str],
    emotions2int: dict[str, int],
) -> pd.DataFrame:
    """
    Compute per-dataset macro F1 (and precision/recall) for each dataset.
    Returns a DataFrame with one row per dataset.
    """
    label_indices = [emotions2int[e] for e in emotions]
    rows = []

    for ds in sorted(np.unique(ds_test)):
        mask = ds_test == ds
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", labels=label_indices
        )

        rows.append(
            {
                "dataset": ds,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "n_samples": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def plot_per_dataset_f1_bar(df_ds: pd.DataFrame, outfile: Path) -> None:
    """Bar plot: dataset vs macro F1."""
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_ds, x="dataset", y="f1_macro")
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1")
    plt.title("Per-dataset macro F1")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[saved per-dataset F1 plot] {outfile}")