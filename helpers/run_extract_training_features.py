"""
run_extract_training_features.py
====================================

Description
-------------------------
End-to-end orchestration script to prepare the **training data** for
speech–emotion modelling. It performs THREE stages in a single call:

    1) Ensure raw emotional speech datasets are available locally
       (download them if necessary, via a helper module).

    2) Assemble a merged CSV manifest of audio files across multiple
       datasets (EMODB, RAVDESS, TESS), with optional filtering
       by gender and emotions.

    3) Extend that merged CSV with acoustic features computed using
       **Librosa** and/or **OpenSMILE**, without any intermediate "path
       remapping". It reads absolute file paths directly from the CSV,
       converts each WAV to mono/16 kHz with ffmpeg, extracts features,
       and concatenates them to the original rows in-place (aligned by
       input order).

Structure
-------------------------------
Datasets handled
^^^^^^^^^^^^^^^^
The following datasets can be downloaded (if missing) into `--raw-data-dir`:

    - RAVDESS : "uwrfkaggler/ravdess-emotional-speech-audio"
    - TESS    : "ejlok1/toronto-emotional-speech-set-tess"
    - EmoDB   : "piyushagni5/berlin-database-of-emotional-speech-emodb"

Downloads are delegated to:

    scripts.download_training_datasets.download_audio.download_audio_files_kaggle

Stage 2: assembling / filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Assembling is delegated to:

    scripts.assembling_training_datasets.assembler

You can:

    - filter by gender (`--gender`)
    - choose which emotions to keep (`--emotions`, IDs or names)

The resulting manifest CSV has at least:

    dataset,path,emotion_name,emotion_code

Stage 3: feature extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^
From the manifest CSV, the script:

1. Converts WAVs to mono/16 kHz with ffmpeg (files saved in `--conversion-dir`).
2. Extracts features (choose via `--engine`):
       - librosa   : 193-dim vector
                     (40 MFCC + 12 chroma + 128 mel + 7 contrast + 6 tonnetz)
       - opensmile : eGeMAPSv02 Functionals
       - both      : horizontally concatenate both sets
                     (OpenSMILE columns will be prefixed 'smile_').
3. Concatenates features to the original manifest DataFrame and writes
   `--out-file`.

How to use with examples
------------------------

Example: download datasets if needed, assemble training manifest for selected
emotions, and extract both Librosa + OpenSMILE features:

    python run.py extract-training-features \
        --gender female \
        --emotions happy fear \
        --manifest data_training/merged.csv \
        --out-file data_training/merged_with_features.csv \
        --engine both

Important options
-----------------

    --raw-data-dir DIR             Where raw datasets should live
                                   (default: <repo>/data_training/raw_data).

    --gender {male,female}         Optional gender filter.

    --emotions ...                 List of emotion IDs or names to keep
                                   (default: all universal IDs used by the
                                   assembler).

    --manifest PATH                Where to write the merged manifest CSV
                                   (default: <repo>/data_training/merged).

    --out-file PATH                Where to write the final features CSV
                                   (default: <repo>/data_training/merged_with_features.csv).

    --engine {librosa|opensmile|both}
                                   Feature engine (default: librosa).

    --conversion-dir DIR           Where to write converted WAVs
                                   (default: <repo>/data_training/converted_16khz).

    --no-validate                  Skip header validation before extraction.

    --ffmpeg PATH                  Path to ffmpeg executable.

    --no-download                  Skip Kaggle dataset download step.

Requirements
------------
- ffmpeg installed and on PATH (or pass a custom path in `--ffmpeg`).
- Python: pandas, numpy, librosa, opensmile, tqdm, soundfile, kagglehub.
- Helper functions are expected in:
      scripts/download_training_datasets/download_audio.py
      scripts/assembling_training_datasets/assembler.py
      scripts/extract_features_training/extract.py
"""

from __future__ import annotations

###############
# Standard libraries
###############
import argparse
import sys
from pathlib import Path
from typing import List

###############
# Add project root to sys.path
###############
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

###############
# Third-party libraries
###############
import pandas as pd

###############
# Project imports
###############
from scripts.download_training_datasets.download_audio import (  # type: ignore
    download_audio_files_kaggle,
)
from scripts.assembling_training_datasets.assembler import (  # type: ignore
    assemble_all_datasets,
    save_manifest,
    normalize_emotion_inputs,
    VALID_GENDERS,
)
from scripts.extract_features_training.extract import (  # type: ignore
    EXPECTED_HEADER,
    convert_to_mono_16khz,
    build_librosa_feature_matrix,
    build_opensmile_feature_matrix,
)
from utils import set_global_seed

###############
# Feature extraction core routine
###############
def extract_and_merge_from_csv(
    in_csv: Path | str,
    out_csv: Path | str,
    validate_header: bool = True,
    engine: str = "librosa",  # "librosa", "opensmile", or "both"
    conversion_dir: Path | str = "converted_16khz",
    ffmpeg_exe: Path | str = "ffmpeg",
) -> pd.DataFrame:
    """
    Extract Librosa and/or OpenSMILE features for the **training datasets**
    using a manifest CSV and write feature-augmented tables.

    The CSV must include:
        dataset, path, emotion_name, emotion_code

    The function:
        1) Converts all WAVs in the CSV to mono / 16 kHz (ffmpeg)
        2) Extracts acoustic features depending on `engine`
        3) Writes one or multiple output CSVs:

           engine = "librosa":
               <out_csv> = base + Librosa features

           engine = "opensmile":
               <out_csv> = base + OpenSMILE features

           engine = "both":
               <stem>_librosa.csv      = base + Librosa
               <stem>_opensmile.csv    = base + smile_* (OpenSMILE prefixed)
               <out_csv>               = base + Librosa + smile_*

    Parameters
    ----------
    in_csv : Path or str
        Input manifest CSV containing absolute WAV file paths.
    out_csv : Path or str
        Main output CSV. When engine="both", this becomes the
        *merged* features CSV.
    validate_header : bool
        If True, ensures that the CSV header matches EXPECTED_HEADER.
    engine : {"librosa", "opensmile", "both"}
        Which feature extractor(s) to run.
    conversion_dir : Path or str
        Location where converted mono/16kHz WAVs are written.
    ffmpeg_exe : Path or str
        ffmpeg executable path.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted features. When engine="both",
        this is the merged features table.
    """
    in_csv = Path(in_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(in_csv)

    if validate_header:
        cols = list(base.columns)
        if cols != EXPECTED_HEADER:
            raise ValueError(
                f"CSV header mismatch.\nExpected: {EXPECTED_HEADER}\nFound:    {cols}"
            )

    if "path" not in base.columns:
        raise ValueError("Input CSV must contain a 'path' column of absolute WAV file paths.")

    converted_paths = convert_to_mono_16khz(
        paths=[Path(p) for p in base["path"].astype(str).tolist()],
        target_dir=conversion_dir,
        ffmpeg_exe=ffmpeg_exe,
    )

    base_reset = base.reset_index(drop=True)
    librosa_df = None
    opensmile_df = None

    stem = out_csv.stem
    suffix = out_csv.suffix or ".csv"

    # Paths for the three possible outputs
    librosa_csv = out_csv if engine == "librosa" else out_csv.with_name(f"{stem}_librosa{suffix}")
    opensmile_csv = out_csv if engine == "opensmile" else out_csv.with_name(f"{stem}_opensmile{suffix}")
    merged_csv = out_csv  # for engine == "both"

    # Librosa features
    if engine in ("librosa", "both"):
        librosa_df = build_librosa_feature_matrix(converted_paths)
        df_librosa = pd.concat([base_reset, librosa_df.reset_index(drop=True)], axis=1)
        df_librosa.to_csv(librosa_csv, index=False, float_format="%.6f")
        print(f"[INFO] Librosa features written to: {librosa_csv.resolve()}")

        if engine == "librosa":
            return df_librosa

    # OpenSMILE features
    if engine in ("opensmile", "both"):
        opensmile_df = build_opensmile_feature_matrix(converted_paths)
        opensmile_pref = opensmile_df.add_prefix("smile_") if engine == "both" else opensmile_df

        df_smile = pd.concat([base_reset, opensmile_pref.reset_index(drop=True)], axis=1)
        df_smile.to_csv(opensmile_csv, index=False, float_format="%.6f")
        print(f"[INFO] OpenSMILE features written to: {opensmile_csv.resolve()}")

        if engine == "opensmile":
            return df_smile

    # Librosa + OpenSMILE
    if engine == "both":
        if librosa_df is None or opensmile_df is None:
            raise RuntimeError("Engine 'both' requires both Librosa and OpenSMILE features.")

        opensmile_pref = opensmile_df.add_prefix("smile_")
        df_merged = pd.concat(
            [
                base_reset,
                librosa_df.reset_index(drop=True),
                opensmile_pref.reset_index(drop=True),
            ],
            axis=1,
        )
        df_merged.to_csv(merged_csv, index=False, float_format="%.6f")
        print(f"[INFO] Merged Librosa+OpenSMILE features written to: {merged_csv.resolve()}")

        return df_merged

    raise ValueError(f"Unknown engine: {engine}")


###############
# Parser
###############
def _parse_args() -> argparse.Namespace:
    def emo_token(x: str):
        try:
            return int(x)
        except ValueError:
            return x.lower()

    parser = argparse.ArgumentParser(
        description=(
            "Download training datasets (if needed), assemble a filtered "
            "training manifest, and extract acoustic features into a single CSV."
        )
    )

    # Stage 1 – raw data location
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=ROOT / "data_training" / "raw_data",
        help=(
            "Directory under which raw Kaggle datasets will be stored "
            "(default: <repo>/data_training/raw_data)."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip datasets download step (only ensure raw_data_dir exists).",
    )

    # Stage 2 – assembling / filtering
    parser.add_argument(
        "--gender",
        type=str,
        choices=sorted(VALID_GENDERS),
        default=None,
        help="Optional gender filter: male or female.",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        type=emo_token,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help=(
            "Emotions to keep; accepts either integer IDs or names. "
            "Default: all universal IDs known by the assembler."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data_training" / "merged.csv",
        help=(
            "Path where the assembled training manifest CSV will be written "
            "(default: <repo>/data_training/merged.csv)."
        ),
    )

    # Stage 3 – feature extraction
    parser.add_argument(
        "--out-file",
        type=Path,
        default=ROOT / "data_training" / "merged_with_features.csv",
        help=(
            "Output CSV path for manifest + features "
            "(default: <repo>/data_training/merged_with_features.csv)."
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["librosa", "opensmile", "both"],
        default="librosa",
        help="Feature extraction engine (default: librosa).",
    )
    parser.add_argument(
        "--conversion-dir",
        type=Path,
        default=ROOT / "data_training" / "converted_16khz",
        help=(
            "Directory to write converted mono/16kHz WAVs "
            "(default: <repo>/data_training/converted_16khz)."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip CSV header validation before feature extraction.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        help="Path to ffmpeg executable (or keep default if on PATH).",
    )

    return parser.parse_args()


###############
# Main
###############
def main() -> None:
    set_global_seed()
    args = _parse_args()

    ### Stage 1 – Ensure / download raw datasets
    ## Prepare raw_data_dir and optionally call Kaggle downloader
    raw_data_dir: Path = args.raw_data_dir
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        print(f"[INFO] Skipping dataset download (raw_data_dir={raw_data_dir}).")
    else:
        print(f"[INFO] Ensuring datasets exist under: {raw_data_dir}")
        download_audio_files_kaggle(local_root=raw_data_dir, download_one=False)

    ### Stage 2 – Assemble training manifest
    ## Filter by gender and emotions, then save manifest CSV
    try:
        normalized_emotions: List[int] = normalize_emotion_inputs(args.emotions)
    except ValueError as e:
        print(f"[ERROR] Invalid emotions specification: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Assembling datasets from: {raw_data_dir}")
    print(f"[INFO] Gender filter: {args.gender or '(none)'}")
    print(f"[INFO] Emotions (normalized universal IDs): {sorted(set(normalized_emotions))}")

    results = assemble_all_datasets(
        dataset_root=raw_data_dir,
        gender=args.gender,
        emotions=normalized_emotions,
    )

    total = sum(len(v) for v in results.values())
    print("\n[SUMMARY – assembly]")
    for k, v in results.items():
        print(f"  - {k:7s}: {len(v)} files")
    print(f"  - {'TOTAL':7s}: {total} files")

    manifest_path = save_manifest(results, args.manifest)
    print(f"\n[INFO] Manifest written to: {manifest_path}")

    ### Stage 3 – Extract features and merge into CSV
    ## Call extract_and_merge_from_csv with chosen engine
    merged = extract_and_merge_from_csv(
        in_csv=manifest_path,
        out_csv=args.out_file,
        engine=args.engine,
        validate_header=not args.no_validate,
        conversion_dir=args.conversion_dir,
        ffmpeg_exe=args.ffmpeg,
    )
    print(f"[INFO] Final dataset with features: {args.out_file} (rows: {len(merged)})")


if __name__ == "__main__":
    main()