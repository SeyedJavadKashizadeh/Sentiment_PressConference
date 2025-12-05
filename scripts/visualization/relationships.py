
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def plot_hyperparameter_search_overview(
    df: pd.DataFrame,
    metric: str,
    hyperparameter: str,
    hue: str,
    style: str,
    title: str,
    xlabel: str,
    ylabel: str,
    savepath: Path,
    target_metric: str = "Accuracy",
) -> None:
    savepath.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    hue_vals = df[hue].unique()
    palette = sns.color_palette("viridis", len(hue_vals))

    sns.lineplot(
        data=df,
        x=hyperparameter,
        y=metric,
        hue=hue,
        style=style,
        markers=True,
        dashes=False,
        ax=ax,
        palette=palette,
        linewidth=2.0,
        markersize=8,
        errorbar=None,
    )

    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Test {ylabel}")

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    sns.despine(ax=ax)
    ax.legend(title=f"{hue} / {style}", frameon=False, fontsize=9)

    outfile = savepath / f"{hyperparameter}_effect_on_test_{target_metric}.png"
    fig.savefig(outfile, dpi=300)
    print(f"[saved] {outfile}")
    plt.close(fig)


def generalization_and_overfitting_analysis(
    df: pd.DataFrame,
    train_metric: str,
    test_metric: str,
    hyperparameter: str,
    title: str,
    savepath: Path,
    target_metric: str = "Accuracy",
) -> None:
    savepath.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))

    x = df[train_metric]
    y = df[test_metric]

    sc = ax.scatter(x, y, c=df[hyperparameter], cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{hyperparameter}")

    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"{train_metric}")
    ax.set_ylabel(f"{test_metric}")
    ax.set_title(f"{title}")

    plt.tight_layout()
    outfile = savepath / f"generalization_{hyperparameter}_{target_metric}.png"
    plt.savefig(outfile, dpi=300)
    print(f"[saved] {outfile}")

    plt.close(fig)


def stability_across_folds_analysis(
    df: pd.DataFrame,
    metric_test: str,
    labels_cols: str,
    savepath: Path,
    top_k: int = 4,
    target_metric: str = "Accuracy",
) -> None:
    savepath.mkdir(parents=True, exist_ok=True)
    topk = df.nlargest(top_k, metric_test)

    base_metric = metric_test.replace("mean_", "")

    fold_cols = [
        col for col in df.columns if col.startswith("split") and base_metric in col
    ]

    if not fold_cols:
        raise ValueError(
            f"No fold columns found for base metric '{base_metric}'. "
            f"Available columns: {list(df.columns)}"
        )

    long_df = topk.melt(
        id_vars=labels_cols,
        value_vars=fold_cols,
        var_name="fold",
        value_name=base_metric,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=long_df,
        x=labels_cols,
        y=base_metric,
        ax=ax,
    )
    sns.stripplot(
        data=long_df,
        x=labels_cols,
        y=base_metric,
        ax=ax,
        color="red",
        size=5,
        jitter=True,
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel(f"{base_metric} per fold")
    ax.set_title(f"Stability across folds for top {top_k} configurations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    outfile = savepath / f"stability_across_folds_{target_metric}.png"
    plt.savefig(outfile, dpi=300)
    print(f"[saved] {outfile}")

    plt.close(fig)


def computational_cost_analysis(
    df: pd.DataFrame,
    time_metric: str,
    test_metric: str,
    main_hp: str,
    secondary_hp: str,
    savepath: Path,
    target_metric: str = "Accuracy",
) -> None:
    savepath.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        df[time_metric],
        df[test_metric],
        c=df[main_hp],
        s=20 + 100 * df[secondary_hp],
        cmap="viridis",
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{main_hp}")

    ax.set_xlabel(f"{time_metric} per CV run [s]")
    ax.set_ylabel(f"{test_metric}")
    ax.set_title(f"{test_metric} vs {time_metric}")

    plt.legend(title=f"{main_hp} / {secondary_hp}", frameon=False, fontsize=9)
    plt.tight_layout()
    outfile = savepath / f"computational_cost_analysis_{target_metric}.png"
    plt.savefig(outfile, dpi=300)
    print(f"[saved] {outfile}")

    plt.close(fig)