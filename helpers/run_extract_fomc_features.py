"""
run_extract_fomc_features.py
=============================

End-to-end orchestration script to prepare the **FOMC audio dataset**
for downstream modelling. It performs THREE stages in a single call:

    1) Download the Federal Reserve FOMC press conference audio dataset
       from Hugging Face (requires authentication via HF_TOKEN).

    2) Convert all WAV files to **mono / 16 kHz** using ffmpeg while
       preserving the dataset’s folder structure.

    3) Extract acoustic features using **Librosa**, **OpenSMILE**, or both,
       and save them as Parquet tables aligned by `(folder, filename)`.

Dataset handled
---------------
This script works with **one dataset**:

    - FedSentimentLab/Fed_audio_text_video  (private HF dataset)

Authentication
--------------
It is assumed that you have a `.env` file in the **repository root**
containing:

    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

This token must have read access to the private dataset. The script loads
it automatically and passes it to Hugging Face API calls.

Stage 1 — Downloading
---------------------
Downloading is delegated to:

    scripts.download_fomc_datasets.download_audio

The script:

    - Lists all available audio folders from the HF dataset
    - Downloads the WAV files into `--download-dir`
    - Preserves the original structure, e.g.:

          download_dir/
              audio_files_split/
                  20110427/
                  20110622/
                  ...

Stage 2 — Conversion to mono / 16 kHz
-------------------------------------
All WAVs are converted with ffmpeg and saved under `--convert-dir`,
mirroring the same folder structure as the downloaded data.

Stage 3 — Feature extraction
----------------------------
For each converted WAV file, the script extracts features using:

    - librosa   (193 dims: MFCC, chroma, mel, contrast, tonnetz)
    - opensmile (eGeMAPSv02 Functionals)
    - both      (concatenation of both sets)

The resulting feature matrices are saved inside `--features-dir` as:

    features_librosa.parquet
    features_opensmile.parquet
    features_merged.parquet   (only when --engine both)

Usage
-----

1) Default settings, Librosa only:

    python helpers/run_extract_fomc_features.py

2) Both engines, specifying all output folders:

    python helpers/run_extract_fomc_features.py \
        --download-dir data_FOMC/audios \
        --convert-dir  data_FOMC/mono_16kHz \
        --features-dir data_FOMC/features \
        --engine both

3) OpenSMILE only:

    python helpers/run_extract_fomc_features.py --engine opensmile

Important options
-----------------

    --download-dir DIR     Where to store raw HF downloads
                           (default: <repo>/data_FOMC/audios)

    --convert-dir DIR      Where converted mono/16kHz WAVs are written
                           (default: <repo>/data_FOMC/mono_16kHz)

    --features-dir DIR     Where Parquet feature tables are written
                           (default: <repo>/data_FOMC/features)

    --engine {librosa|opensmile|both}
                           Feature backend (default: librosa)

    --ffmpeg PATH          Path to ffmpeg executable
                           (default: "ffmpeg")

Requirements
------------

- ffmpeg installed and on PATH (or specify with --ffmpeg)
- .env file containing HF_TOKEN
- Python: pandas, numpy, librosa, opensmile, tqdm, soundfile,
          huggingface_hub, python-dotenv

Notes
-----

- The dataset is private; **HF_TOKEN is mandatory**.
- Folder structure from Hugging Face is always preserved.
- Extraction proceeds file-by-file; failures skip the file but continue.
- Feature extraction logic is implemented in:

        scripts/extract_features_FOMC/extract.py

"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.download_fomc_datasets.download_audio import (
    get_audio_files_by_folder,
    download_audio_files_hf,
)
from scripts.extract_features_FOMC.extract import (
    convert_to_mono_16khz,
    build_feature_dataframe,
    build_features_df_opensmile,
)

def load_hf_token():
    token_file = Path(__file__).resolve().parents[1] / ".env"
    token = token_file.read_text().strip()
    if token.startswith("HF_TOKEN="):
        token = token.split("=", 1)[1]   # remove the prefix
    return token

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments controlling I/O directories and feature engine.

    Returns
    -------
    argparse.Namespace
        Namespace with attributes:
            download_dir : Path or None
            convert_dir  : Path or None
            features_dir : Path or None
            engine       : str in {"librosa", "opensmile", "both"}
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download, convert, and extract acoustic features from the "
            "FedSentimentLab/Fed_audio_text_video FOMC audio dataset."
        )
    )

    parser.add_argument(
        "--download-dir",
        type=Path,
        default=ROOT / "data_fomc" / "raw_data",
        help=(
            "Directory where raw audio files from Hugging Face will be stored. "
            "Defaults to <repo>/data_FOMC/raw_data"
        ),
    )

    parser.add_argument(
        "--convert-dir",
        type=Path,
        default=ROOT / "data_fomc" / "converted_16khz",
        help=(
            "Directory where converted mono/16 kHz WAVs will be written. "
            "Defaults to <repo>/data_FOMC/converted_16khz."
        ),
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=ROOT / "data_fomc" / "features",
        help=(
            "Directory where feature Parquet files will be stored. "
            "Defaults to <repo>/data_FOMC/features."
        ),
    )

    parser.add_argument(
        "--engine",
        choices=["librosa", "opensmile", "both"],
        default="librosa",
        help=(
            "Feature extraction engine: 'librosa', 'opensmile', or 'both'. "
            "Default: 'librosa'."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve directories (CLI overrides defaults)
    download_dir = args.download_dir
    converted_dir = args.convert_dir 
    features_dir = args.features_dir 

    download_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    engine = args.engine

    HF_TOKEN = load_hf_token()

    # List files on HF
    files_by_folder = get_audio_files_by_folder(HF_TOKEN)


    # Download into download_dir
    local_paths = download_audio_files_hf(
        hf_token=HF_TOKEN,
        files_by_folder=files_by_folder,
        local_root=download_dir,
        download_one=False,  # or True if you only want the first folder
    )
    # Convert to mono 16kHz into converted_dir
    converted_paths = convert_to_mono_16khz(
        local_paths=local_paths,
        target_root=converted_dir,
        ffmpeg_exe = rf"C:\ffmpeg\bin\ffmpeg.exe"
    )

    df_librosa = None
    df_smile = None

    # Librosa features
    if engine in ("librosa", "both"):
        df_librosa = build_feature_dataframe(converted_paths)
        df_librosa.to_parquet(features_dir / "features_librosa.parquet")
        print("Saved Librosa features:", features_dir / "features_librosa.parquet")

    # OpenSMILE features
    if engine in ("opensmile", "both"):
        df_smile = build_features_df_opensmile(converted_paths)
        df_smile.to_parquet(features_dir / "features_opensmile.parquet")
        print("Saved OpenSMILE features:", features_dir / "features_opensmile.parquet")

    # Librosa + OpenSMILE features (only if both were computed)
    if engine == "both" and df_librosa is not None and df_smile is not None:
        df_merged = df_librosa.join(df_smile, how="inner")
        df_merged.to_parquet(features_dir / "features_merged.parquet")
        print("Saved merged features:", features_dir / "features_merged.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
