"""
run_eda_analysis.py
===================

Description
-----------
End-to-end exploratory data analysis for the **training features**.

Given:
    - a Librosa feature CSV
    - an OpenSMILE feature CSV

this script produces:

1) Emotion distribution plots (all emotions + subset).
2) Librosa summary-feature vs emotion tables + z-score heatmaps.
3) OpenSMILE key-feature vs emotion tables + z-score heatmaps.
4) t-SNE embeddings for Librosa and OpenSMILE features:
       - colored by emotion
       - colored by dataset

All figures and tables are automatically saved under --save-dir.

Typical usage
-------------
    python run.py eda-analysis

or with custom paths:

    python run.py eda-analysis \
        --librosa-csv  data_training/merged_with_features_librosa.csv \
        --opensmile-csv data_training/merged_with_features_opensmile.csv \
        --save-dir     outputs/eda_analysis
"""

###################
# Standard libraries
###################
import argparse
from pathlib import Path

###################
# Third-party libraries
###################
import pandas as pd

###################
# Project root and imports
###################
ROOT = Path(__file__).resolve().parents[1]

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eda_register.features_class_librosa import (  # type: ignore
    MFCC_SUMMARY,
    MEL_SUMMARY,
    CHROMA_SUMMARY,
    CONTRAST_SUMMARY,
    TONNETZ_SUMMARY,
)
from scripts.eda_register.features_class_opensmile import (  # type: ignore
    OPENSMILE_KEY_FEATURES_RENAMED,
)

from scripts.visualization.eda import (
    plot_emotion_distribution,
    plot_emotion_feature_zscore_heatmap,
    build_emotion_feature_table,
    tsne_embeddings_plots
)

from utils import set_global_seed

###################
# Block feature builders
###################
def add_librosa_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add aggregated (summary) Librosa features to the input DataFrame.

    Expects per-band features:
        mfcc_0..39, chroma_0..11, mel_0..127, contrast_0..6, tonnetz_0..5.
    Creates summary columns:
        - MFCC_SUMMARY:    mfcc_mean, mfcc_std
        - MEL_SUMMARY:     mel_low_mean, mel_mid_mean, mel_high_mean
        - CHROMA_SUMMARY:  chroma_mean, chroma_std
        - CONTRAST_SUMMARY: contrast_mean
        - TONNETZ_SUMMARY: tonnetz_mean
    """
    df = df.copy()

    # MFCC
    mfcc_cols = [f"mfcc_{i}" for i in range(40) if f"mfcc_{i}" in df.columns]
    if len(mfcc_cols) == 40:
        df["mfcc_mean"] = df[mfcc_cols].mean(axis=1)
        df["mfcc_std"] = df[mfcc_cols].std(axis=1)
    else:
        print("[warn] Missing some mfcc_0..mfcc_39; skipping MFCC summaries.")

    # MEL
    mel_cols_all = [f"mel_{i}" for i in range(128) if f"mel_{i}" in df.columns]
    if len(mel_cols_all) == 128:
        low_band = [f"mel_{i}" for i in range(0, 20)]
        mid_band = [f"mel_{i}" for i in range(20, 60)]
        high_band = [f"mel_{i}" for i in range(60, 128)]

        df["mel_low_mean"] = df[low_band].mean(axis=1)
        df["mel_mid_mean"] = df[mid_band].mean(axis=1)
        df["mel_high_mean"] = df[high_band].mean(axis=1)
    else:
        print("[warn] Missing some mel_0..mel_127; skipping MEL summaries.")

    # CHROMA
    chroma_cols = [f"chroma_{i}" for i in range(12) if f"chroma_{i}" in df.columns]
    if len(chroma_cols) == 12:
        df["chroma_mean"] = df[chroma_cols].mean(axis=1)
        df["chroma_std"] = df[chroma_cols].std(axis=1)
    else:
        print("[warn] Missing some chroma_0..chroma_11; skipping CHROMA summaries.")

    # CONTRAST
    contrast_cols = [f"contrast_{i}" for i in range(7) if f"contrast_{i}" in df.columns]
    if len(contrast_cols) == 7:
        df["contrast_mean"] = df[contrast_cols].mean(axis=1)
    else:
        print("[warn] Missing some contrast_0..contrast_6; skipping CONTRAST summaries.")

    # TONNETZ
    tonnetz_cols = [f"tonnetz_{i}" for i in range(6) if f"tonnetz_{i}" in df.columns]
    if len(tonnetz_cols) == 6:
        df["tonnetz_mean"] = df[tonnetz_cols].mean(axis=1)
    else:
        print("[warn] Missing some tonnetz_0..tonnetz_5; skipping TONNETZ summaries.")

    return df

###################
# Argument parsing
###################
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA on Librosa and OpenSMILE training features."
    )

    parser.add_argument(
        "--librosa-csv",
        type=Path,
        default=ROOT / "data_training" / "merged_with_features_librosa.csv",
        help="Librosa training features CSV.",
    )
    parser.add_argument(
        "--opensmile-csv",
        type=Path,
        default=ROOT / "data_training" / "merged_with_features_opensmile.csv",
        help="OpenSMILE training features CSV.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=ROOT / "outputs" / "eda_analysis",
        help="Base directory where EDA outputs will be stored.",
    )

    return parser.parse_args()


###################
# Main
###################
def main() -> None:
    set_global_seed()
    args = _parse_args()

    save_root: Path = args.save_dir
    save_root.mkdir(parents=True, exist_ok=True)

    # Load data
    librosa_df = pd.read_csv(args.librosa_csv)
    opensmile_df = pd.read_csv(args.opensmile_csv)

    # 1) Emotion distribution
    out = save_root / "emotion_distribution"
    plot_emotion_distribution(librosa_df, out)
    emotions_subset = ["happy", "neutral", "sad", "angry"]
    plot_emotion_distribution(librosa_df, out, emotions_subset)

    # 2) Librosa: feature–emotion relationships
    df_librosa_summary = add_librosa_summary_features(librosa_df)
    out_librosa = save_root / "librosa_feature_relationships"
    out_librosa.mkdir(parents=True, exist_ok=True)

    librosa_features = (
        MFCC_SUMMARY + MEL_SUMMARY + CHROMA_SUMMARY + CONTRAST_SUMMARY + TONNETZ_SUMMARY
    )

    table_librosa = build_emotion_feature_table(
        df_librosa_summary,
        feature_list=librosa_features,
        emotion_col="emotion_name",
    )
    table_librosa.to_csv(out_librosa / "librosa_feature_emotion_table.csv")

    plot_emotion_feature_zscore_heatmap(
        df=df_librosa_summary,
        feature_list=librosa_features,
        savepath=out_librosa,
        emotion_col="emotion_name",
        rowwise_normalize=True,
        annot=True,
        title="Librosa summary features: z-score by emotion",
        filename="librosa_zscore_heatmap.png",
    )

    # 3) OpenSMILE: feature–emotion relationships
    out_smile = save_root / "opensmile_feature_relationships"
    out_smile.mkdir(parents=True, exist_ok=True)

    opensmile_features = list(OPENSMILE_KEY_FEATURES_RENAMED.keys())

    table_smile = build_emotion_feature_table(
        opensmile_df,
        feature_list=opensmile_features,
        rename_map=OPENSMILE_KEY_FEATURES_RENAMED,
        emotion_col="emotion_name",
    )
    table_smile.to_csv(out_smile / "opensmile_feature_emotion_table.csv")

    plot_emotion_feature_zscore_heatmap(
        df=opensmile_df,
        feature_list=opensmile_features,
        rename_map=OPENSMILE_KEY_FEATURES_RENAMED,
        savepath=out_smile,
        emotion_col="emotion_name",
        rowwise_normalize=True,
        annot=True,
        title="OpenSMILE key features: z-score by emotion",
        filename="opensmile_zscore_heatmap.png",
    )

    # 4) t-SNE for Librosa and OpenSMILE
    tsne_root = save_root / "tsne"
    features_librosa = librosa_df.columns[4:]
    _ = tsne_embeddings_plots(
        df=librosa_df,
        feature_cols=features_librosa,
        savepath=tsne_root / "tsne_librosa",
    )

    features_opensmile = opensmile_df.columns[4:]
    _ = tsne_embeddings_plots(
        df=opensmile_df,
        feature_cols=features_opensmile,
        savepath=tsne_root / "tsne_opensmile",
    )

if __name__ == "__main__":
    main()
