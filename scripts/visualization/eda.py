import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pathlib import Path
from typing import List, Optional, Sequence

sns.set_theme(style="whitegrid")

def plot_emotion_distribution(
    data: pd.DataFrame,
    savepath: Path,
    emotions: Optional[List[str]] = None,
) -> None:
    """Plot the distribution of emotions in the dataset and save PNG."""
    savepath.mkdir(parents=True, exist_ok=True)

    if emotions is not None:
        data = data[data["emotion_name"].isin(emotions)]
        emotions = sorted(emotions)
        name = "subset_"
    else:
        emotions = sorted(data["emotion_name"].unique())
        name = "all_"

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=data,
        x="emotion_name",
        order=data["emotion_name"].value_counts().index,
    )
    plt.title(f"Distribution of Emotions: {', '.join(emotions)}")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    outfile = savepath / f"{name}emotion_distribution.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[saved] {outfile}")


def build_emotion_feature_table(
    df: pd.DataFrame,
    feature_list: List[str],
    rename_map: Optional[dict[str, str]] = None,
    emotion_col: str = "emotion_name",
) -> pd.DataFrame:
    """
    Build a table where rows = features, columns = emotions,
    and values = 'mean (std)' for that feature within that emotion.
    """
    emotions = sorted(df[emotion_col].unique())
    table = pd.DataFrame(index=feature_list, columns=emotions)

    for feature in feature_list:
        if feature not in df.columns:
            print(f"[warn] Feature {feature} missing, skipping.")
            continue

        for emotion in emotions:
            values = df.loc[df[emotion_col] == emotion, feature].to_numpy()
            mean = np.mean(values)
            std = np.std(values)
            table.loc[feature, emotion] = f"{mean:.3f} ({std:.3f})"

    if rename_map is not None:
        valid_map = {k: v for k, v in rename_map.items() if k in table.index}
        table = table.rename(index=valid_map)

    return table


def build_emotion_feature_zscore_table(
    df: pd.DataFrame,
    feature_list: List[str],
    rename_map: Optional[dict[str, str]] = None,
    emotion_col: str = "emotion_name",
) -> pd.DataFrame:
    """
    Compute a table of z-score differences:
    rows = features, columns = emotions,
    values = z-score of mean(feature | emotion).

    z = (mean_emotion - mean_global) / std_global
    """
    emotions = sorted(df[emotion_col].unique())
    ztable = pd.DataFrame(index=feature_list, columns=emotions, dtype=float)

    for feature in feature_list:
        if feature not in df.columns:
            print(f"[warn] Feature {feature} missing, skipping in z-score table.")
            continue

        global_mean = df[feature].mean()
        global_std = df[feature].std()

        if global_std == 0 or np.isnan(global_std):
            print(f"[warn] Feature {feature} has zero or NaN std; skipping.")
            continue

        for emotion in emotions:
            values = df.loc[df[emotion_col] == emotion, feature]
            mean_emotion = values.mean()
            z = (mean_emotion - global_mean) / global_std
            ztable.loc[feature, emotion] = z

    if rename_map is not None:
        valid_map = {k: v for k, v in rename_map.items() if k in ztable.index}
        ztable = ztable.rename(index=valid_map)

    return ztable


def normalize_ztable_rowwise(ztable: pd.DataFrame) -> pd.DataFrame:
    """Normalize each feature row by its maximum absolute z-score."""
    z_absmax = ztable.abs().max(axis=1).replace(0, np.nan)
    z_norm = ztable.div(z_absmax, axis=0)
    return z_norm


def plot_emotion_feature_zscore_heatmap(
    df: pd.DataFrame,
    feature_list: List[str],
    savepath: Path,
    rename_map: Optional[dict[str, str]] = None,
    emotion_col: str = "emotion_name",
    rowwise_normalize: bool = False,
    annot: bool = True,
    title: str = "Emotion-wise z-score differences per feature",
    filename: str = "emotion_feature_zscore_heatmap.png",
) -> None:
    """Build and plot a heatmap of emotion-wise z-scores."""
    savepath.mkdir(parents=True, exist_ok=True)

    ztable = build_emotion_feature_zscore_table(
        df=df,
        feature_list=feature_list,
        rename_map=rename_map,
        emotion_col=emotion_col,
    )

    if rowwise_normalize:
        ztable_plot = normalize_ztable_rowwise(ztable)
    else:
        ztable_plot = ztable.copy()

    ztable_plot = ztable_plot.dropna(how="all", axis=0)
    ztable_plot = ztable_plot.dropna(how="all", axis=1)

    plt.figure(figsize=(12, max(4, 0.4 * len(ztable_plot.index))))
    sns.set_theme(style="white")

    sns.heatmap(
        ztable_plot.astype(float),
        cmap="coolwarm",
        center=0.0,
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "z-score (per feature)"},
    )

    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Feature")
    plt.tight_layout()

    outfile = savepath / filename
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[saved] {outfile}")


def tsne_embeddings_plots(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    savepath: Path,
    emotion_col: str = "emotion_name",
    dataset_col: str = "dataset",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute a 2D t-SNE embedding and generate:
        - tsne_by_emotion.png
        - tsne_by_dataset.png
    """
    savepath.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns found in DataFrame.")

    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    df_embed = df.copy()
    df_embed["tsne_1"] = X_tsne[:, 0]
    df_embed["tsne_2"] = X_tsne[:, 1]

    sns.set_theme(style="white")

    # By emotion
    if emotion_col in df_embed.columns:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df_embed,
            x="tsne_1",
            y="tsne_2",
            hue=emotion_col,
            alpha=0.7,
            s=10,
            edgecolor="none",
        )
        plt.title("t-SNE embedding colored by emotion")
        plt.tight_layout()
        outfile = savepath / "tsne_by_emotion.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"[saved] {outfile}")
    else:
        print(f"[warn] Column '{emotion_col}' not found; skipping emotion plot.")

    # By dataset
    if dataset_col in df_embed.columns:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df_embed,
            x="tsne_1",
            y="tsne_2",
            hue=dataset_col,
            alpha=0.7,
            s=10,
            edgecolor="none",
        )
        plt.title("t-SNE embedding colored by dataset")
        plt.tight_layout()
        outfile = savepath / "tsne_by_dataset.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"[saved] {outfile}")
    else:
        print(f"[warn] Column '{dataset_col}' not found; skipping dataset plot.")

    return df_embed
