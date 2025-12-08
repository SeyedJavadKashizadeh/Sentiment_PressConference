"""
run_extract_fomc_features.py
=============================

Description
-------------------------
End-to-end orchestration script to prepare the **FOMC audio dataset**
for downstream modelling. It performs THREE stages in a single call:

    1) Download the Federal Reserve FOMC press conference audio dataset
       from Hugging Face (requires authentication via HF_TOKEN).

    2) Convert all WAV files to **mono / 16 kHz** using ffmpeg while
       preserving the dataset’s folder structure.

    3) Extract acoustic features using **Librosa**, **OpenSMILE**, or both,
       and save them as Parquet tables aligned by `(folder, filename)`.

Structure
-------------------------------
Dataset handled
^^^^^^^^^^^^^^^
This script works with **one dataset**:

    - FedSentimentLab/Fed_audio_text_video  (private HF dataset)

Authentication
^^^^^^^^^^^^^^
It is assumed that you have a `.env` file in the **repository root**
containing:

    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

This token must have read access to the private dataset. The script loads
it automatically and passes it to Hugging Face API calls.

Stage 1 — Downloading
^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All WAVs are converted with ffmpeg and saved under `--convert-dir`,
mirroring the same folder structure as the downloaded data.

Stage 3 — Feature extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For each converted WAV file, the script extracts features using:

    - librosa   (193 dims: MFCC, chroma, mel, contrast, tonnetz)
    - opensmile (eGeMAPSv02 Functionals)
    - both      (concatenation of both sets)

The resulting feature matrices are saved inside `--features-dir` as:

    features_librosa.parquet
    features_opensmile.parquet
    features_merged.parquet   (only when --engine both)

How to use with examples
------------------------

1) Default settings, Librosa only:

    python run.py extract-fomc-features

2) Both engines, specifying all output folders:

    python run.py extract-fomc-features \
        --download-dir data_fomc/raw_data \
        --convert-dir  data_fomc/converted_16khz \
        --features-dir data_fomc/features \
        --engine both

3) OpenSMILE only:

    python run.py extract-fomc-features --engine opensmile

Important options
-----------------

    --download-dir DIR     Where to store raw HF downloads
                           (default: <repo>/data_fomc/raw_data)

    --convert-dir DIR      Where converted mono/16kHz WAVs are written
                           (default: <repo>/data_fomc/converted_16khz)

    --features-dir DIR     Where Parquet feature tables are written
                           (default: <repo>/data_fomc/features)

    --engine {librosa|opensmile|both}
                           Feature backend (default: librosa)

    --ffmpeg PATH          Path to ffmpeg executable
                           (default: C:\\ffmpeg\\bin\\ffmpeg.exe)

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

###############
# Standard libraries
###############
import argparse
from pathlib import Path
import sys

###############
# Add project root to sys.path
###############
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

###############
# Project imports
###############
from scripts.download_fomc_datasets.download_audio import (  # type: ignore
    get_audio_files_by_folder,
    download_audio_files_hf,
)
from scripts.extract_features_FOMC.extract import (  # type: ignore
    convert_to_mono_16khz,
    build_feature_dataframe,
    build_features_df_opensmile,
)
from utils import set_global_seed
###############
# HF token loader
###############
def load_hf_token() -> str:
    """
    Load HF_TOKEN from a .env file at the repository root.

    Expected format (at least one line containing):
        HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env file with HF_TOKEN not found at: {env_path}")

    token: str | None = None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip()
            break

    if not token:
        raise RuntimeError("HF_TOKEN not found in .env file.")

    return token


###############
# Argument parsing
###############
def _parse_args() -> argparse.Namespace:
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
            "Defaults to <repo>/data_fomc/raw_data."
        ),
    )

    parser.add_argument(
        "--convert-dir",
        type=Path,
        default=ROOT / "data_fomc" / "converted_16khz",
        help=(
            "Directory where converted mono/16 kHz WAVs will be written. "
            "Defaults to <repo>/data_fomc/converted_16khz."
        ),
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=ROOT / "data_fomc" / "features",
        help=(
            "Directory where feature Parquet files will be stored. "
            "Defaults to <repo>/data_fomc/features."
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

    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        help="Path to ffmpeg executable (default: C:\\ffmpeg\\bin\\ffmpeg.exe).",
    )

    return parser.parse_args()


###############
# Main
###############
def main() -> None:
    set_global_seed()
    args = _parse_args()

    ### Prepare directories
    ## Ensure all target folders exist
    download_dir = args.download_dir
    converted_dir = args.convert_dir
    features_dir = args.features_dir

    download_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    engine = args.engine
    ffmpeg_exe = args.ffmpeg

    ### Load HF token and list files
    ## Authentication + HF listing
    HF_TOKEN = load_hf_token()

    files_by_folder = get_audio_files_by_folder(HF_TOKEN)

    ### Download audio files
    ## Preserve HF folder structure under download_dir
    local_paths = download_audio_files_hf(
        hf_token=HF_TOKEN,
        files_by_folder=files_by_folder,
        local_root=download_dir,
        download_one=False,  # set True if you want to limit to one folder
    )

    ### Convert to mono 16 kHz (Main step)
    ## Call shared conversion utility
    converted_paths = convert_to_mono_16khz(
        local_paths=local_paths,
        target_root=converted_dir,
        ffmpeg_exe=str(ffmpeg_exe),
    )

    df_librosa = None
    df_smile = None

    ### Librosa features
    ## Extract and save as Parquet if requested
    if engine in ("librosa", "both"):
        df_librosa = build_feature_dataframe(converted_paths)
        out_librosa = features_dir / "features_librosa.parquet"
        df_librosa.to_parquet(out_librosa)
        print(f"Saved Librosa features: {out_librosa.resolve()}")

    ### OpenSMILE features
    ## Extract and save as Parquet if requested
    if engine in ("opensmile", "both"):
        df_smile = build_features_df_opensmile(converted_paths)
        out_smile = features_dir / "features_opensmile.parquet"
        df_smile.to_parquet(out_smile)
        print(f"Saved OpenSMILE features: {out_smile.resolve()}")

    ### Merged features
    ## Join Librosa + OpenSMILE on common index when engine == "both"
    if engine == "both" and df_librosa is not None and df_smile is not None:
        df_merged = df_librosa.join(df_smile, how="inner")
        out_merged = features_dir / "features_merged.parquet"
        df_merged.to_parquet(out_merged)
        print(f"Saved merged features: {out_merged.resolve()}")

    print("Done.")


if __name__ == "__main__":
    main()
