"""
scripts/extract_features_FOMC/extract.py
=================================================
Audio conversion and feature extraction utilities for the **FOMC dataset**.

This module provides:
    • Conversion of downloaded FOMC WAV files to mono / 16 kHz using ffmpeg
      while preserving their relative paths (e.g. audio_files_split/20110427/...).

    • Extraction of acoustic features:
          - Librosa (193-dim vector: MFCC, chroma, mel, contrast, tonnetz),
            returned as a MultiIndex DataFrame (folder, filename).
          - OpenSMILE (eGeMAPSv02 Functionals), also as a MultiIndex DataFrame.

All functions keep indices aligned with the original relative paths so that
the resulting tables can be joined or merged cleanly in higher-level scripts
(e.g. `run_extract_fomc_features.py`).
"""


from typing import Dict

from pathlib import Path
import os
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import librosa
import opensmile


def convert_to_mono_16khz(
    local_paths: Dict[str, Path],
    target_root: Path,
    ffmpeg_exe: Path | str = rf"C:\ffmpeg\bin\ffmpeg.exe"
) -> Dict[str, Path]:
    """
    Convert audio files to mono 16 kHz while preserving their relative paths.

    Parameters
    ----------
    local_paths : dict
        Mapping {relative_path: absolute_input_path} from the download step.
        Example key:  'audio_files_split/20110427/file.wav'
        Example value: '/your/data_raw/audio_files_split/20110427/file.wav'
    target_root : Path
        Root directory where converted files will be stored. The keys'
        relative paths are preserved under this directory.
    ffmpeg_exe : Path | str
        Root directory where the ffmpeg.exe lies on. Please look at readme

    Returns
    -------
    dict
        Mapping {relative_path: converted_absolute_path}.
    """
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    converted: Dict[str, Path] = {}

    for rel_path, abs_path in local_paths.items():
        target_path = (target_root / rel_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            command = [
                str(ffmpeg_exe),
                "-y",
                "-i", str(abs_path),
                "-ac", "1",
                "-ar", "16000",
                str(target_path),
            ]
            try:
                subprocess.run(command, check=False)
            except Exception as e:
                print("Command:", command)
                print(f"Error type: {type(e).__name__}")
                print(f"Error: {e}")
                raise
            print(f"Converted {abs_path} -> {target_path}")
        else:
            print(f"Already exists: {target_path}")

        converted[rel_path] = target_path
    
    return converted


def extract_features_librosa(file_name: Path) -> np.ndarray:
    """
    Extract Librosa audio features:

        - 40 MFCCs (based on 128 Mel bands)
        - 12 Chroma coefficients
        - 128 Mel-spectrogram bands
        - 7 Spectral contrast coefficients
        - 6 Tonnetz coefficients

    Returns
    -------
    np.ndarray
        Array of shape (1, n_features).
    """
    file_name = Path(file_name)

    with sf.SoundFile(str(file_name)) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(
        librosa.feature.mfcc(
            y=X,
            sr=sample_rate,
            n_mfcc=40,
            n_mels=128,
            fmin=0,
            fmax=8000,
        ).T,
        axis=0,
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=X,
            sr=sample_rate,
            n_chroma=12,
            n_fft=4096,
        ).T,
        axis=0,
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=X,
            sr=sample_rate,
            n_mels=128,
            fmin=0,
            fmax=8000,
        ).T,
        axis=0,
    )

    contrast = np.mean(
        librosa.feature.spectral_contrast(
            S=stft,
            sr=sample_rate,
            fmin=200.0,
        ).T,
        axis=0,
    )

    tonnetz = np.mean(
        librosa.feature.tonnetz(
            y=librosa.effects.harmonic(X),
            sr=sample_rate,
        ).T,
        axis=0,
    )

    features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
    return features.reshape(1, -1)


def build_feature_dataframe(converted_paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Build a MultiIndex DataFrame of Librosa features.

    Parameters
    ----------
    converted_paths : dict
        Mapping {relative_path: converted_absolute_path} from convert_to_mono_16khz.

    Returns
    -------
    pd.DataFrame
        MultiIndex index: (folder, filename), columns: Librosa features.
    """
    records = []
    index_tuples = []

    for rel_path, abs_path in tqdm(converted_paths.items(), desc="Extracting Librosa features"):
        # rel_path example: 'audio_files_split/20110427/FILE.wav'
        parts = rel_path.split("/")
        folder = parts[1] if len(parts) > 1 else ""
        filename = os.path.basename(str(abs_path))

        try:
            features = extract_features_librosa(abs_path)
            records.append(features.flatten())
            index_tuples.append((folder, filename))
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    if not records:
        return pd.DataFrame()

    feature_array = np.vstack(records)
    df = pd.DataFrame(feature_array)

    mfcc_cols = [f"mfcc_{i}" for i in range(40)]
    chroma_cols = [f"chroma_{i}" for i in range(12)]
    mel_cols = [f"mel_{i}" for i in range(128)]
    contrast_cols = [f"contrast_{i}" for i in range(7)]
    tonnetz_cols = [f"tonnetz_{i}" for i in range(6)]

    df.columns = mfcc_cols + chroma_cols + mel_cols + contrast_cols + tonnetz_cols
    df.index = pd.MultiIndex.from_tuples(index_tuples, names=["folder", "filename"])

    return df


def build_features_df_opensmile(converted_paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Extract OpenSMILE eGeMAPSv02 features for all audio files and build
    a MultiIndex DataFrame.

    Parameters
    ----------
    converted_paths : dict
        Mapping {relative_path: converted_absolute_path} from convert_to_mono_16khz.

    Returns
    -------
    pd.DataFrame
        MultiIndex index: (folder, filename), columns: OpenSMILE features.
    """
    feature_rows = []
    index_tuples = []

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    for rel_path, abs_path in tqdm(converted_paths.items(), desc="Extracting OpenSMILE features"):
        parts = rel_path.split("/")
        folder = parts[1] if len(parts) > 1 else ""
        filename = os.path.basename(str(abs_path))

        try:
            features_df = smile.process_file(str(abs_path))
            if len(features_df) == 1:
                features_row = features_df.iloc[0]
            else:
                features_row = features_df.mean(axis=0)
            feature_rows.append(features_row)
            index_tuples.append((folder, filename))
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    if not feature_rows:
        return pd.DataFrame()

    combined_df = pd.DataFrame(feature_rows)
    combined_df.index = pd.MultiIndex.from_tuples(index_tuples, names=["folder", "filename"])

    return combined_df
