"""
Evaluation utilities to replicate and compare metrics used in the Voice of Monetary Policy (VOMP) paper.

This module centralizes the loading, computation, comparison, and plotting of
the "voice tone" metric derived from predicted emotion labels, following the
definitions used in the VOMP paper.

It supports:
- Evaluating one prediction CSV against VOMP reference data
- Evaluating multiple models (multiple prediction CSVs)
- Exporting per-date merged comparisons and aggregated metric tables
- Plotting VOMP (paper) vs multiple model predictions on the same figures

It exposes:

    • subset_VOMP_timeframes(...)
        Convenience filter to restrict data to the VOMP sample window.

    • load_predictions(...), load_VOMP_tones(...), load_VOMP_classifications(...)
        CSV loaders for model predictions and VOMP reference datasets.

    • calculate_voice_tone(...)
        Implements the VOMP-style voice tone:
            voice_tone = (positive - negative) / (positive + negative)
        with positive = happy + pleasant_surprise and negative = sad + angry.

    • compare_predictions_to_VOMP(...)
        Merges predicted per-date tone with VOMP per-date data and produces
        aggregated comparison metrics for All and (if available) by Chair.

    • eval_one_model(...), eval_many_models(...)
        Evaluation entry points for one or multiple prediction CSVs.

    • save_results(...), plot_multi_models(...)
        Output helpers for CSV exports and multi-model plots.

Typical usage
-------------
Designed to be called from a thin runner that parses CLI arguments:

>>> from pathlib import Path
>>> from scripts.vomp_eval import eval_many_models, save_results, plot_multi_models
>>>
>>> results = eval_many_models(
...     pred_csvs=[Path("baseline_preds.csv"), Path("adv_preds.csv")],
...     vomp_dir=Path("vomp_data"),
...     model_names=["baseline", "advanced_ravdess_tess"],
... )
>>> for name, (merged, metrics, tone) in results.items():
...     save_results(merged, metrics, Path("outputs/vomp_eval") / name)
>>> plot_multi_models(results, Path("outputs/vomp_eval/plots"))

Assumptions
-----------
- Prediction CSVs contain at least columns: `item` and `emotion`.
- The `item` column encodes the press conference date as YYYYMMDD and a
  filename including a chair identifier, following the regex in `load_predictions`.
- VOMP tone CSV contains `Press conference date` and typically includes:
    `Speaker`, `Positive responses`, `Neutral responses`, `Negative responses`.
  A `Tone` column may or may not exist. If absent, we still compare via
  VOMP aggregate metrics, and plots use mean lines instead of a VOMP time series.
- VOMP classifications CSV contains a `Metric` column which becomes the index,
  and group columns like: "All (1)", "Bernanke (2)", "Yellen (3)", "Powell (4)".
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
# Constants
###############################################################################

VOMP_START_DATE = datetime.datetime(2011, 1, 26)
VOMP_END_DATE = datetime.datetime(2019, 7, 31)

POSITIVE_EMOTIONS = ("happy", "pleasant_surprise")
NEGATIVE_EMOTIONS = ("sad", "angry")
NEUTRAL_EMOTIONS = ("neutral",)

VOMP_TONES_FILENAME = "VOMP_tones.csv"
VOMP_CLASSIFICATIONS_FILENAME = "VOMP_classifications.csv"


###############################################################################
# Timeframe utilities
###############################################################################

def subset_VOMP_timeframes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset a DataFrame to the canonical VOMP sample window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame restricted to [2011-01-26, 2019-07-31] inclusive.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "subset_VOMP_timeframes expects a DataFrame indexed by pd.DatetimeIndex."
        )

    mask = (df.index >= VOMP_START_DATE) & (df.index <= VOMP_END_DATE)
    return df.loc[mask]


###############################################################################
# CSV loaders
###############################################################################

def load_predictions(pred_file: Path) -> pd.DataFrame:
    """
    Load model emotion predictions from a CSV file.

    The function expects at least columns:
        - item
        - emotion

    It extracts a `date` from the `item` field using a regex that matches
    common project conventions such as:
        ('20110126', 'CHAIR_BERNANKE_s12.wav')

    Parameters
    ----------
    pred_file : Path
        Path to the predictions CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `date` (DatetimeIndex) with at least:
            - emotion
            - filename (if extractable)
            - chairman (if extractable)
    """
    raw_df = pd.read_csv(pred_file)

    required = {"item", "emotion"}
    missing = required.difference(set(raw_df.columns))
    if missing:
        raise ValueError(f"{pred_file} missing required columns: {sorted(missing)}")

    extracted = raw_df["item"].astype(str).str.extract(
        r"\(?'(\d{8})', '((?:CHAIR(?:MAN)?)_([A-Z]+)_s\d+\.wav)'\)?"
    )
    raw_df["date"] = pd.to_datetime(extracted[0], format="%Y%m%d", errors="coerce")
    raw_df["filename"] = extracted[1]
    raw_df["chairman"] = extracted[2]

    raw_df = raw_df.dropna(subset=["date"])
    raw_df = raw_df.set_index("date").sort_index()

    return raw_df


def load_VOMP_tones(vomp_tones_file: Path) -> pd.DataFrame:
    """
    Load VOMP per-press-conference tone data.

    Parameters
    ----------
    vomp_tones_file : Path
        Path to the VOMP tones CSV. Expected to include:
            - Press conference date (parseable as datetime)
        Typically also includes:
            - Speaker
            - Positive responses / Neutral responses / Negative responses
            - Tone (optional)

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by press conference date.
    """
    df = pd.read_csv(vomp_tones_file, parse_dates=["Press conference date"])
    df = df.rename(columns={"Press conference date": "date"})
    df = df.set_index("date").sort_index()
    return df


def load_VOMP_classifications(vomp_class_file: Path) -> pd.DataFrame:
    """
    Load VOMP aggregated classification metrics.

    Parameters
    ----------
    vomp_class_file : Path
        Path to the VOMP classifications CSV. Must contain a `Metric` column
        which is used as the row index.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Metric, with columns for groups such as:
        "All (1)", "Bernanke (2)", "Yellen (3)", "Powell (4)".
    """
    df = pd.read_csv(vomp_class_file)
    if "Metric" not in df.columns:
        raise ValueError(f"{vomp_class_file} must contain a 'Metric' column.")
    return df.set_index("Metric")


###############################################################################
# Metric computations
###############################################################################

def calculate_voice_tone(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-date voice tone from predicted emotion labels.

    Following the VOMP definition:
        voice_tone = (positive - negative) / (positive + negative)

    where:
        positive = count(happy) + count(pleasant_surprise)
        negative = count(sad) + count(angry)

    Notes
    -----
    - Neutral labels are counted separately but do not enter the denominator.
    - If (positive + negative) == 0 for a date, the resulting tone is set to 0.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame indexed by date, with an `emotion` column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date containing:
            - voice_tone
            - positive_count
            - negative_count
            - neutral_count
    """
    if "emotion" not in predictions.columns:
        raise ValueError("calculate_voice_tone expects an 'emotion' column in predictions.")

    daily_counts = predictions.groupby(predictions.index).agg(
        positive_count=("emotion", lambda s: s.isin(POSITIVE_EMOTIONS).sum()),
        negative_count=("emotion", lambda s: s.isin(NEGATIVE_EMOTIONS).sum()),
        neutral_count=("emotion", lambda s: s.isin(NEUTRAL_EMOTIONS).sum()),
    )

    numerator = daily_counts["positive_count"] - daily_counts["negative_count"]
    denominator = daily_counts["positive_count"] + daily_counts["negative_count"]

    voice_tone = numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([voice_tone.rename("voice_tone"), daily_counts], axis=1)


def compare_predictions_to_VOMP(
    tone_metric: pd.DataFrame,
    vomp_classifications: pd.DataFrame,
    vomp_tones: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare predicted tone metrics to VOMP reference metrics.

    This performs:
      1) An inner join on date between predicted per-date tone and VOMP per-date data.
      2) A computation of aggregated counts/ratios and voice tone mean/std for:
         - All (overall)
         - Each chair (Bernanke / Yellen / Powell) if `Speaker` is available.

    Parameters
    ----------
    tone_metric : pd.DataFrame
        Output of `calculate_voice_tone`, indexed by date.
    vomp_classifications : pd.DataFrame
        Aggregated VOMP classification table indexed by Metric.
    vomp_tones : pd.DataFrame
        Per-date VOMP tones table indexed by date (press conference date).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        merged_df_tone : pd.DataFrame
            Per-date merged table containing predicted and VOMP fields.
        metrics_df : pd.DataFrame
            Aggregated comparison metrics with columns such as:
                - All_vomp, All_pred
                - Bernanke_vomp, Bernanke_pred (if available)
                - Yellen_vomp, Yellen_pred (if available)
                - Powell_vomp, Powell_pred (if available)
    """
    merged = tone_metric.merge(vomp_tones, left_index=True, right_index=True, how="inner")

    # Normalize predicted column names.
    merged = merged.rename(
        columns={
            "voice_tone": "voice_tone_pred",
            "positive_count": "positive_count_pred",
            "negative_count": "negative_count_pred",
            "neutral_count": "neutral_count_pred",
        }
    )

    # Normalize VOMP column names if present.
    merged = merged.rename(
        columns={
            "Positive responses": "positive_count_vomp",
            "Neutral responses": "neutral_count_vomp",
            "Negative responses": "negative_count_vomp",
            "Tone": "tone_vomp",
        }
    )

    metrics_df = pd.DataFrame(
        index=[
            "Positive (count)",
            "Negative (count)",
            "Neutral (count)",
            "Voice tone mean",
            "Voice tone standard deviation",
            "Positive ratio",
            "Negative ratio",
            "Neutral ratio",
        ]
    )

    def safe_ratio(a: float, b: float) -> float:
        return float(a / b) if b and b > 0 else 0.0

    # -------------------------------------------------------------------------
    # All (overall)
    # -------------------------------------------------------------------------
    all_pos_vomp = float(vomp_classifications.loc["Positive (count)", "All (1)"])
    all_neg_vomp = float(vomp_classifications.loc["Negative (count)", "All (1)"])
    all_neu_vomp = float(vomp_classifications.loc["Neutral (count)", "All (1)"])
    all_total_vomp = all_pos_vomp + all_neg_vomp + all_neu_vomp

    all_pos_pred = float(merged["positive_count_pred"].sum())
    all_neg_pred = float(merged["negative_count_pred"].sum())
    all_neu_pred = float(merged["neutral_count_pred"].sum())
    all_total_pred = all_pos_pred + all_neg_pred + all_neu_pred

    metrics_df["All_vomp"] = [
        all_pos_vomp,
        all_neg_vomp,
        all_neu_vomp,
        float(vomp_classifications.loc["Voice tone mean", "All (1)"]),
        float(vomp_classifications.loc["Voice tone standard deviation", "All (1)"]),
        safe_ratio(all_pos_vomp, all_total_vomp),
        safe_ratio(all_neg_vomp, all_total_vomp),
        safe_ratio(all_neu_vomp, all_total_vomp),
    ]

    metrics_df["All_pred"] = [
        all_pos_pred,
        all_neg_pred,
        all_neu_pred,
        float(merged["voice_tone_pred"].mean()),
        float(merged["voice_tone_pred"].std()),
        safe_ratio(all_pos_pred, all_total_pred),
        safe_ratio(all_neg_pred, all_total_pred),
        safe_ratio(all_neu_pred, all_total_pred),
    ]

    # -------------------------------------------------------------------------
    # By chair (requires `Speaker` in per-date VOMP file)
    # -------------------------------------------------------------------------
    if "Speaker" in merged.columns:
        chair_map = {
            "Bernanke": "Bernanke (2)",
            "Yellen": "Yellen (3)",
            "Powell": "Powell (4)",
        }

        for chair, col in chair_map.items():
            chair_data = merged[merged["Speaker"] == chair]

            chair_pos_vomp = float(vomp_classifications.loc["Positive (count)", col])
            chair_neg_vomp = float(vomp_classifications.loc["Negative (count)", col])
            chair_neu_vomp = float(vomp_classifications.loc["Neutral (count)", col])
            chair_total_vomp = chair_pos_vomp + chair_neg_vomp + chair_neu_vomp

            chair_pos_pred = float(chair_data["positive_count_pred"].sum())
            chair_neg_pred = float(chair_data["negative_count_pred"].sum())
            chair_neu_pred = float(chair_data["neutral_count_pred"].sum())
            chair_total_pred = chair_pos_pred + chair_neg_pred + chair_neu_pred

            metrics_df[f"{chair}_vomp"] = [
                chair_pos_vomp,
                chair_neg_vomp,
                chair_neu_vomp,
                float(vomp_classifications.loc["Voice tone mean", col]),
                float(vomp_classifications.loc["Voice tone standard deviation", col]),
                safe_ratio(chair_pos_vomp, chair_total_vomp),
                safe_ratio(chair_neg_vomp, chair_total_vomp),
                safe_ratio(chair_neu_vomp, chair_total_vomp),
            ]

            metrics_df[f"{chair}_pred"] = [
                chair_pos_pred,
                chair_neg_pred,
                chair_neu_pred,
                float(chair_data["voice_tone_pred"].mean()) if len(chair_data) else 0.0,
                float(chair_data["voice_tone_pred"].std()) if len(chair_data) else 0.0,
                safe_ratio(chair_pos_pred, chair_total_pred),
                safe_ratio(chair_neg_pred, chair_total_pred),
                safe_ratio(chair_neu_pred, chair_total_pred),
            ]

    return merged, metrics_df


###############################################################################
# Output helpers
###############################################################################

def save_results(merged_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save per-date merged comparison output and aggregated metrics to CSV files.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Per-date merged output from `compare_predictions_to_VOMP`.
    metrics_df : pd.DataFrame
        Aggregated metrics output from `compare_predictions_to_VOMP`.
    out_dir : Path
        Directory where outputs are written. Created if missing.

    Returns
    -------
    None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_dir / "tone_comp_grading.csv", index=True)
    metrics_df.to_csv(out_dir / "tone_comp_metrics.csv", index=True)


###############################################################################
# Evaluation API
###############################################################################

def eval_one_model(
    pred_csv: Path,
    vomp_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one model's predictions against VOMP metrics.

    Parameters
    ----------
    pred_csv : Path
        Path to the model predictions CSV.
    vomp_dir : Path
        Directory containing:
            - VOMP_tones.csv
            - VOMP_classifications.csv

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        merged_df_tone : pd.DataFrame
            Per-date merged table (pred vs VOMP).
        metrics_df : pd.DataFrame
            Aggregated comparison metrics (All and possibly by chair).
        tone_metric_pred : pd.DataFrame
            Raw per-date predicted tone table returned by `calculate_voice_tone`.
    """
    vomp_tones = load_VOMP_tones(vomp_dir / VOMP_TONES_FILENAME)
    vomp_class = load_VOMP_classifications(vomp_dir / VOMP_CLASSIFICATIONS_FILENAME)

    preds = load_predictions(pred_csv)
    preds = subset_VOMP_timeframes(preds)

    tone_metric = calculate_voice_tone(preds)
    merged_df, metrics_df = compare_predictions_to_VOMP(tone_metric, vomp_class, vomp_tones)

    return merged_df, metrics_df, tone_metric


def eval_many_models(
    pred_csvs: List[Path],
    vomp_dir: Path,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Evaluate multiple prediction CSVs.

    Parameters
    ----------
    pred_csvs : list[Path]
        List of prediction CSV paths.
    vomp_dir : Path
        Directory containing VOMP reference files.
    model_names : list[str] | None
        Optional list of model names. If provided, must match `len(pred_csvs)`.
        If omitted, uses `pred_csv.stem` for each model.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Mapping model_name -> (merged_df, metrics_df, tone_metric_pred).
    """
    if model_names is not None and len(model_names) != len(pred_csvs):
        raise ValueError("If provided, model_names must have same length as pred_csvs.")

    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    for i, pred_csv in enumerate(pred_csvs):
        name = model_names[i] if model_names is not None else pred_csv.stem
        merged_df, metrics_df, tone_metric = eval_one_model(pred_csv, vomp_dir)
        results[name] = (merged_df, metrics_df, tone_metric)

    return results


###############################################################################
# Plotting
###############################################################################

def plot_multi_models(
    model_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    out_dir: Path,
) -> None:
    """
    Generate comparison plots across models AND against VOMP (paper) reference metrics.

    Outputs (PNG) written into `out_dir`:

    - plot_voice_tone_timeseries_models_vs_vomp.png
        Overlays model predicted tone time series.
        If a per-date VOMP `Tone` column exists, overlays it as well.
        Otherwise, overlays the VOMP mean tone as a horizontal dashed line.

    - plot_all_voice_tone_mean_models_vs_vomp.png
        Bar chart comparing "Voice tone mean" for All_vomp vs All_pred of each model.

    - plot_all_ratios_models_vs_vomp.png
        Grouped bars comparing positive/negative/neutral ratios for All_vomp vs each model's All_pred.

    Parameters
    ----------
    model_results : dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Mapping model_name -> (merged_df, metrics_df, tone_metric_pred).
    out_dir : Path
        Directory where plots are written.

    Returns
    -------
    None
    """
    if not model_results:
        raise ValueError("plot_multi_models received an empty model_results mapping.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use any one model to extract VOMP reference aggregates (identical across models).
    any_merged, any_metrics, _ = next(iter(model_results.values()))

    # -------------------------------------------------------------------------
    # 1) Time series overlay: models + VOMP reference
    # -------------------------------------------------------------------------
    plt.figure()

    # Prefer true VOMP per-date tone if available (Tone -> tone_vomp).
    if "tone_vomp" in any_merged.columns:
        s_vomp = any_merged["tone_vomp"].sort_index()
        plt.plot(s_vomp.index, s_vomp.values, label="VOMP (paper)", linewidth=2.0)
    else:
        # Fallback: horizontal line at VOMP aggregate mean
        if "All_vomp" in any_metrics.columns and "Voice tone mean" in any_metrics.index:
            vomp_mean = float(any_metrics.loc["Voice tone mean", "All_vomp"])
            plt.axhline(vomp_mean, label="VOMP mean (paper)", linestyle="--")

    # Plot each model predicted per-date tone
    for name, (merged, _, tone_metric) in model_results.items():
        if "voice_tone_pred" in merged.columns:
            s = merged["voice_tone_pred"].sort_index()
        else:
            # Fallback: if merged doesn't have voice_tone_pred, use raw tone table
            s = tone_metric["voice_tone"].sort_index()
        plt.plot(s.index, s.values, label=name)

    plt.title("Voice tone over time: models vs VOMP reference")
    plt.xlabel("Date")
    plt.ylabel("Voice tone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_voice_tone_timeseries_models_vs_vomp.png", dpi=200)
    plt.close()

    # Helper for safe access into metrics frames
    def get_metric(metrics_df: pd.DataFrame, row: str, col: str) -> float:
        if row in metrics_df.index and col in metrics_df.columns:
            return float(metrics_df.loc[row, col])
        return 0.0

    names = list(model_results.keys())

    # -------------------------------------------------------------------------
    # 2) Bar: voice tone mean (All_vomp vs All_pred)
    # -------------------------------------------------------------------------
    vomp_mean = get_metric(any_metrics, "Voice tone mean", "All_vomp")
    model_means = [get_metric(model_results[n][1], "Voice tone mean", "All_pred") for n in names]

    plt.figure()
    plt.bar(["VOMP (paper)"] + names, [vomp_mean] + model_means)
    plt.title("All: voice tone mean (VOMP paper vs models)")
    plt.ylabel("Mean")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_all_voice_tone_mean_models_vs_vomp.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # 3) Grouped bars: ratios (All_vomp vs All_pred)
    # -------------------------------------------------------------------------
    vomp_pos = get_metric(any_metrics, "Positive ratio", "All_vomp")
    vomp_neg = get_metric(any_metrics, "Negative ratio", "All_vomp")
    vomp_neu = get_metric(any_metrics, "Neutral ratio", "All_vomp")

    pos = [get_metric(model_results[n][1], "Positive ratio", "All_pred") for n in names]
    neg = [get_metric(model_results[n][1], "Negative ratio", "All_pred") for n in names]
    neu = [get_metric(model_results[n][1], "Neutral ratio", "All_pred") for n in names]

    labels = ["VOMP (paper)"] + names
    pos_all = [vomp_pos] + pos
    neg_all = [vomp_neg] + neg
    neu_all = [vomp_neu] + neu

    x = np.arange(len(labels))
    width = 0.25

    plt.figure()
    plt.bar(x - width, pos_all, width, label="Positive")
    plt.bar(x, neg_all, width, label="Negative")
    plt.bar(x + width, neu_all, width, label="Neutral")

    plt.title("All: emotion ratios (VOMP paper vs models)")
    plt.ylabel("Ratio")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_all_ratios_models_vs_vomp.png", dpi=200)
    plt.close()
