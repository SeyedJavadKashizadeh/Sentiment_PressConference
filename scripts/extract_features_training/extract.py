from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional, Dict

import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa
import opensmile


###############################################################################
# Constants
###############################################################################
EXPECTED_HEADER: List[str] = ["dataset", "path", "emotion_name", "emotion_code"]


###############################################################################
# Conversion
###############################################################################

def convert_to_mono_16khz(
    paths: Sequence[Path | str],
    target_dir: Path | str,
    ffmpeg_exe: Path | str = "ffmpeg",
) -> List[Optional[Path]]:
    """
    Convert each input WAV to mono/16 kHz using ffmpeg, preserving **input order**.

    Parameters
    ----------
    paths : Sequence[Path | str]
        Absolute paths to source WAV files (same order as CSV).
    target_dir : Path | str
        Directory where converted files are written (flat; names are prefixed with the row index).
    ffmpeg_exe : Path | str
        ffmpeg executable (e.g. r\"C:\\ffmpeg\\bin\\ffmpeg.exe\" on Windows) or 'ffmpeg' if on PATH.

    Returns
    -------
    List[Optional[Path]]
        Absolute paths to converted files in the same order as `paths`.
        If a conversion fails, that position is `None` (feature rows become NaN).
    """
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    converted: List[Optional[Path]] = []

    for i, src in enumerate(paths):
        src = Path(src)
        out_name = f"{i:06d}__{src.name}"
        out_path = (target_dir / out_name).resolve()

        if not out_path.exists():
            cmd = [
                str(ffmpeg_exe), "-y",
                "-i", str(src),
                "-ac", "1",
                "-ar", "16000",
                str(out_path),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"[ffmpeg error] {src}\n{e.stderr.decode(errors='ignore')}")
                converted.append(None)
                continue

        converted.append(out_path)

    return converted


###############################################################################
# Feature extraction Librosa
###############################################################################

def _librosa_feature_column_names() -> List[str]:
    mfcc_cols = [f"mfcc_{i}" for i in range(40)]
    chroma_cols = [f"chroma_{i}" for i in range(12)]
    mel_cols = [f"mel_{i}" for i in range(128)]
    contrast_cols = [f"contrast_{i}" for i in range(7)]
    tonnetz_cols = [f"tonnetz_{i}" for i in range(6)]
    return mfcc_cols + chroma_cols + mel_cols + contrast_cols + tonnetz_cols  # 193


def _extract_features_librosa_single(file_path: Path) -> np.ndarray:
    """
    Returns a (193,) float32 vector. Assumes WAV is already mono @ 16 kHz.
    """
    with sf.SoundFile(str(file_path)) as sfh:
        X = sfh.read(dtype="float32")
        sr = sfh.samplerate

    stft = np.abs(librosa.stft(X))

    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40, n_mels=128, fmin=0, fmax=8000).T.mean(axis=0)
    chroma = librosa.feature.chroma_stft(y=X, sr=sr, n_chroma=12, n_fft=4096).T.mean(axis=0)
    mel = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=128, fmin=0, fmax=8000).T.mean(axis=0)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=200.0).T.mean(axis=0)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T.mean(axis=0)

    feats = np.concatenate([mfccs, chroma, mel, contrast, tonnetz]).astype(np.float32)
    return feats


def build_librosa_feature_matrix(paths: Sequence[Optional[Path]]) -> pd.DataFrame:
    """
    Given a list of converted WAV paths (some entries may be None if conversion failed),
    return a DataFrame of shape (N, 193) aligned with input order.

    Rows with missing/failed audio get NaNs.
    """
    cols = _librosa_feature_column_names()
    out = np.full((len(paths), len(cols)), np.nan, dtype=np.float32)

    for i, p in enumerate(tqdm(paths, desc="Extracting features (librosa)")):
        if p is None or not Path(p).exists():
            continue
        try:
            out[i, :] = _extract_features_librosa_single(Path(p))
        except Exception as e:
            print(f"[librosa error] {p}: {e}")
            # leave NaNs

    return pd.DataFrame(out, columns=cols)


###############################################################################
# OpenSmile
###############################################################################

def build_opensmile_feature_matrix(paths: Sequence[Optional[Path]]) -> pd.DataFrame:
    """
    One row per input path (aligned order). Rows that failed are NaN.

    Returns a DataFrame with OpenSMILE (eGeMAPSv02 Functionals) columns.
    """
    if opensmile is None:
        raise ImportError("opensmile is not installed. `pip install opensmile` to use --engine opensmile.")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows: List[pd.Series] = []
    for i, p in enumerate(tqdm(paths, desc="Extracting features (OpenSMILE)")):
        if p is None or not Path(p).exists():
            rows.append(pd.Series(dtype="float32"))  # placeholder
            continue
        try:
            df = smile.process_file(str(p))  # 1-row DF
            rows.append(df.iloc[0])
        except Exception as e:
            print(f"[OpenSMILE error] {p}: {e}")
            rows.append(pd.Series(dtype="float32"))

    # Build a DataFrame, align lengths; outer join over all columns then reindex in order
    feat_df = pd.DataFrame(rows)
    return feat_df
