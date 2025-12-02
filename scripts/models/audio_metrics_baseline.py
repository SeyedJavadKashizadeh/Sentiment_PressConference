# A file to replicate the mertrics used by Voice of monetary policy paper
import os
from pathlib import Path

# Ensure TensorFlow uses legacy tf.keras behavior (if needed in your environment).
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import pandas as pd
import numpy as np
from typing import List
#import tensorflow as tf
#from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
#from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
import datetime
from pathlib import Path


def subset_VOMP_timeframes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset the DataFrame to include only entries within the Voice of Monetary Policy (VOMP)
    timeframes, specifically from January 26, 2011 to July 31, 2019.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime index.
    last_FOMC : datetime
        The date of the last FOMC meeting.

    Returns
    -------
    pd.DataFrame
        Subset DataFrame containing only entries within the specified timeframe.
    """
    start_date = datetime.datetime(2011, 1, 26)
    last_date = datetime.datetime(2019, 7, 31)
    mask = (df.index >= start_date) & (df.index <= last_date)
    return df.loc[mask]

def load_predictions(pred_file: Path) -> pd.DataFrame:
    """
    Load predictions from a CSV file.

    Parameters
    ----------
    pred_file : Path
        Path to the CSV file containing predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'item' and 'emotion' columns.
    """
    raw_df = pd.read_csv(pred_file)
    #extract: Press conference date, CHairman, file name from item column
    extracted_data = raw_df['item'].str.extract(r"\(?'(\d{8})', '((?:CHAIR(?:MAN)?)_([A-Z]+)_s\d+\.wav)'\)?")
    raw_df['date'] = pd.to_datetime(extracted_data[0])
    raw_df['filename'] = extracted_data[1]
    raw_df['chairman'] = extracted_data[2]
    raw_df.set_index('date', inplace=True)
    return raw_df
def load_VOMP_tones(vomp_tones_file: Path) -> pd.DataFrame:
    """
    Load VOMP tone data from a CSV file.

    Parameters
    ----------
    vomp_file : Path
        Path to the CSV file containing VOMP tone data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date' and 'tone' columns.
    """
    df = pd.read_csv(vomp_tones_file, parse_dates=['Press conference date'])
    df = df.rename(columns={'Press conference date': 'date', 'Tone': 'tone'})
    df.set_index('date', inplace=True)
    return df

def load_VOMP_classifications(vomp_class_file: Path) -> pd.DataFrame:
    """
    Load VOMP classification data from a CSV file.

    Parameters
    ----------
    vomp_class_file : Path
        Path to the CSV file containing VOMP classification data.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date' and 'classification' columns.
    """
    df = pd.read_csv(vomp_class_file)
    df.set_index('Metric', inplace=True)
    return df



def calculate_voice_tone(predictions: pd.DataFrame):
    """
    following the Voice of monetary policy paper to calculate the tone metric
    as voice_tone = (positive - negative)/(positive + negative)
    where positive = happy + pleasant_surprise
    negative = sad + angry
    
    """
    # Group by date (index) and count positive/negative emotions for each group
    daily_counts = predictions.groupby(predictions.index).agg(
        positive_count=('emotion', lambda s: s.isin(['happy', 'pleasant_surprise']).sum()),
        negative_count=('emotion', lambda s: s.isin(['sad', 'angry']).sum()),
        neutral_count=('emotion', lambda s: s.isin(['neutral']).sum())
    )

    # Calculate the voice tone for each date
    numerator = daily_counts['positive_count'] - daily_counts['negative_count']
    denominator = daily_counts['positive_count'] + daily_counts['negative_count']

    # Avoid division by zero, fill resulting NaNs with 0
    voice_tone = numerator.div(denominator).fillna(0)
    voice_tone = voice_tone.to_frame(name='voice_tone')
    #add each count to the dataframe
    voice_tone = pd.concat([voice_tone, daily_counts], axis=1)
    # Return a DataFrame with the calculated tone for each date
    return voice_tone


def compare_predictions_to_VOMP(tone_metric: pd.DataFrame, vomp_classifications: pd.DataFrame, vomp_tones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare model predictions to VOMP classifications.

    Parameters
    ----------
    tone_metric : pd.DataFrame
        DataFrame with calculated tone metrics from predictions.
    vomp_classifications : pd.DataFrame
        DataFrame with VOMP classifications (rows: metrics, columns: All, Bernanke, Yellen, Powell).
    vomp_tones : pd.DataFrame
        DataFrame with VOMP tone data (indexed by date, contains 'Speaker' column).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Merged DataFrame for comparison and aggregated metrics DataFrame.
    """
    # Merge tone metric with VOMP tones to get speaker information
    merged_df_tone = tone_metric.merge(vomp_tones, left_index=True, right_index=True, how='inner')
    merged_df_tone = merged_df_tone.rename(columns={
        'voice_tone': 'voice_tone_pred', 
        'positive_count': 'positive_count_pred', 
        'negative_count': 'negative_count_pred', 
        'neutral_count': 'neutral_count_pred',
        'Positive responses': 'positive_count_vomp', 
        'Neutral responses': 'neutral_count_vomp',
        'Negative responses': 'negative_count_vomp', 
        'Tone': 'tone_vomp'
    })
    
    # Create aggregated metrics DataFrame matching VOMP classifications structure
    metrics_df = pd.DataFrame(index=['Positive (count)', 'Negative (count)', 'Neutral (count)', 
                                      'Voice tone mean', 'Voice tone standard devi',
                                      'Positive ratio', 'Negative ratio', 'Neutral ratio'])
    
    # Calculate metrics for All (overall)
    all_pos_vomp = vomp_classifications.loc['Positive (count)', 'All (1)']
    all_neg_vomp = vomp_classifications.loc['Negative (count)', 'All (1)']
    all_neu_vomp = vomp_classifications.loc['Neutral (count)', 'All (1)']
    all_total_vomp = all_pos_vomp + all_neg_vomp + all_neu_vomp
    
    all_pos_pred = merged_df_tone['positive_count_pred'].sum()
    all_neg_pred = merged_df_tone['negative_count_pred'].sum()
    all_neu_pred = merged_df_tone['neutral_count_pred'].sum()
    all_total_pred = all_pos_pred + all_neg_pred + all_neu_pred
    
    metrics_df['All_vomp'] = [
        all_pos_vomp,
        all_neg_vomp,
        all_neu_vomp,
        vomp_classifications.loc['Voice tone mean', 'All (1)'],
        vomp_classifications.loc['Voice tone standard deviation', 'All (1)'],
        all_pos_vomp / all_total_vomp if all_total_vomp > 0 else 0,
        all_neg_vomp / all_total_vomp if all_total_vomp > 0 else 0,
        all_neu_vomp / all_total_vomp if all_total_vomp > 0 else 0
    ]
    
    metrics_df['All_pred'] = [
        all_pos_pred,
        all_neg_pred,
        all_neu_pred,
        merged_df_tone['voice_tone_pred'].mean(),
        merged_df_tone['voice_tone_pred'].std(),
        all_pos_pred / all_total_pred if all_total_pred > 0 else 0,
        all_neg_pred / all_total_pred if all_total_pred > 0 else 0,
        all_neu_pred / all_total_pred if all_total_pred > 0 else 0
    ]
    
    # Calculate metrics for each chairman
    for chairman, vomp_col in [('Bernanke', 'Bernanke (2)'), ('Yellen', 'Yellen (3)'), ('Powell', 'Powell (4)')]:
        chairman_data = merged_df_tone[merged_df_tone['Speaker'] == chairman]
        
        chair_pos_vomp = vomp_classifications.loc['Positive (count)', vomp_col]
        chair_neg_vomp = vomp_classifications.loc['Negative (count)', vomp_col]
        chair_neu_vomp = vomp_classifications.loc['Neutral (count)', vomp_col]
        chair_total_vomp = chair_pos_vomp + chair_neg_vomp + chair_neu_vomp
        
        chair_pos_pred = chairman_data['positive_count_pred'].sum()
        chair_neg_pred = chairman_data['negative_count_pred'].sum()
        chair_neu_pred = chairman_data['neutral_count_pred'].sum()
        chair_total_pred = chair_pos_pred + chair_neg_pred + chair_neu_pred
        
        metrics_df[f'{chairman.capitalize()}_vomp'] = [
            chair_pos_vomp,
            chair_neg_vomp,
            chair_neu_vomp,
            vomp_classifications.loc['Voice tone mean', vomp_col],
            vomp_classifications.loc['Voice tone standard deviation', vomp_col],
            chair_pos_vomp / chair_total_vomp if chair_total_vomp > 0 else 0,
            chair_neg_vomp / chair_total_vomp if chair_total_vomp > 0 else 0,
            chair_neu_vomp / chair_total_vomp if chair_total_vomp > 0 else 0
        ]
        
        metrics_df[f'{chairman.capitalize()}_pred'] = [
            chair_pos_pred,
            chair_neg_pred,
            chair_neu_pred,
            chairman_data['voice_tone_pred'].mean() if len(chairman_data) > 0 else 0,
            chairman_data['voice_tone_pred'].std() if len(chairman_data) > 0 else 0,
            chair_pos_pred / chair_total_pred if chair_total_pred > 0 else 0,
            chair_neg_pred / chair_total_pred if chair_total_pred > 0 else 0,
            chair_neu_pred / chair_total_pred if chair_total_pred > 0 else 0
        ]
    
    return merged_df_tone, metrics_df

def save_results(merged_df: pd.DataFrame, metrics_df: pd.DataFrame, outfile: Path) -> None:
    """
    Save the grading DataFrame to a CSV file.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame containing the grading information.
    outfile : Path
        Path to the output CSV file.
    """
    out_file_tone = outfile / "tone_comp_grading.csv"
    merged_df.to_csv(out_file_tone, index=True)

    out_file_metrics = outfile / "tone_comp_metrics.csv"
    metrics_df.to_csv(out_file_metrics, index=True)


def eval_tone_metric(pred_file: Path, VOMP_tones_file: Path, VOMP_class_file: Path) -> tuple:
    """
    Evaluate the tone metric from predictions in a CSV file.

    Parameters
    ----------
    pred_file : str
        Path to the CSV file containing predictions.

    Returns
    -------
    float
        Calculated tone metric.
    """
    # Load VOMP tone data
    vomp_tones = load_VOMP_tones(VOMP_tones_file)
    # Load VOMP classification data
    vomp_classifications = load_VOMP_classifications(VOMP_class_file)

    # Load predictions
    predictions = load_predictions(pred_file)
    #keep only entries within VOMP timeframes
    predictions = subset_VOMP_timeframes(predictions)
    
    # Calculate tone metric
    tone_metric_pred = calculate_voice_tone(predictions)

    merged_df_tone, metrics_df = compare_predictions_to_VOMP(tone_metric_pred, vomp_classifications, vomp_tones)

    return merged_df_tone, metrics_df, tone_metric_pred


def main():
    cur_path = Path(__file__).resolve().parent
    data_dir = cur_path.parent.parent / "data"
    pred_file = data_dir / "predictions_librosa.csv"
    VOMP_tones_file = data_dir / "VOMP_tones.csv"
    VOMP_class_file = data_dir / "VOMP_classifications.csv"
    outfile = data_dir
    merged_df_tone, metrics_df, tone_metric = eval_tone_metric(pred_file, VOMP_tones_file, VOMP_class_file)
    print(f"Tone Metric: {tone_metric}")
    save_results(merged_df_tone, metrics_df, outfile)

if __name__ == "__main__":
    main()