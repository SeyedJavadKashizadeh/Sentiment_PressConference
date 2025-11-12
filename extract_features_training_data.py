"""
extract_features_training_data.py
=================================

High-level CLI to extend a merged audio dataset CSV with acoustic features
computed using **Librosa** and/or **OpenSMILE**, without any intermediate
"path relmapping". It reads absolute file paths directly from the CSV, converts
each WAV to mono/16 kHz with ffmpeg, extracts features, and concatenates them
to the original rows in-place (aligned by input order).

Expected CSV format
-------------------
Columns (order enforced unless --no-validate is used):

    dataset,path,emotion_name,emotion_code

where:
    • dataset      – source dataset name (e.g., "emodb", "ravdess")
    • path         – absolute filesystem path to a WAV file
    • emotion_name – human-readable label (e.g., "happy", "sad")
    • emotion_code – integer/standardized emotion ID

Workflow
--------
1) Load CSV and optionally validate header.
2) Convert WAVs to mono/16 kHz with ffmpeg (files saved in --conversion_dir).
3) Extract features (choose via --engine):
       - librosa: 193-dim vector (40 MFCC + 12 chroma + 128 mel + 7 contrast + 6 tonnetz)
       - opensmile: eGeMAPSv02 Functionals
       - both: horizontally concatenate both sets (OpenSMILE columns will be prefixed 'smile_').
4) Concatenate features to the original DataFrame and write --out_file.

Usage
-----
Example: extract both Librosa + OpenSMILE

    python extract_features_training_data.py \
        --in_file merged.csv \
        --out_file merged_with_features.csv \
        --engine both

Options:
    --engine {librosa|opensmile|both}   Feature engine (default: librosa)
    --conversion_dir DIR                Where to write converted WAVs (default: converted_16khz)
    --no-validate                       Skip header validation

Requirements
------------
- ffmpeg installed and on PATH (or pass a custom path in your extract.py if needed)
- Python: pandas, numpy, librosa, opensmile, tqdm, soundfile

Notes
-----
- This script assumes the helper functions below exist in:
      scripts/extract_features_training/extract.py

      EXPECTED_HEADER
      convert_to_mono_16khz(paths, target_dir, ffmpeg_exe="ffmpeg") -> List[Path]
      build_librosa_feature_matrix(paths: Sequence[Optional[Path]]) -> pd.DataFrame
      build_opensmile_feature_matrix(paths: Sequence[Optional[Path]]) -> pd.DataFrame

- The returned feature DataFrames must be aligned row-by-row with the input list
  of converted paths; any failures should leave NaNs for those rows.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from scripts.extract_features_training.extract import (
    EXPECTED_HEADER,
    convert_to_mono_16khz,
    build_librosa_feature_matrix,
    build_opensmile_feature_matrix,
)

###############################################################################
# Core routine
###############################################################################

def extract_and_merge_from_csv(
    in_csv: Path | str,
    out_csv: Path | str,
    validate_header: bool = True,
    engine: str = "librosa",              # "librosa", "opensmile", or "both"
    conversion_dir: Path | str = "converted_16khz",
    ffmpeg_exe: Path | str = "ffmpeg",
) -> pd.DataFrame:
    """
    Load a merged dataset CSV, convert WAVs to mono/16 kHz, extract acoustic features,
    and save the resulting dataset with appended features.

    Parameters
    ----------
    in_csv : Path | str
        Path to the input CSV file. Must contain:
        ['dataset', 'path', 'emotion_name', 'emotion_code'].
    out_csv : Path | str
        Path to the output CSV file to write. Parent directories will be created if missing.
    validate_header : bool, default=True
        Whether to verify that the CSV header matches EXPECTED_HEADER exactly.
    engine : {"librosa", "opensmile", "both"}, default="librosa"
        Which feature extraction engine(s) to use.
        - "librosa": spectral features (MFCC, chroma, mel, etc.)
        - "opensmile": eGeMAPSv02 functional features
        - "both": concatenate both sets of features
    conversion_dir : Path | str, default="converted_16khz"
        Directory to store the converted mono/16 kHz WAVs.
    ffmpeg_exe : Path | str, default="ffmpeg"
        Path to the ffmpeg executable (e.g. "ffmpeg" or "C:/ffmpeg/bin/ffmpeg.exe").

    Returns
    -------
    pd.DataFrame
        The merged DataFrame containing the original metadata columns and the extracted features.
    """
    # Resolve paths
    in_csv = Path(in_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    base = pd.read_csv(in_csv)

    # Optional header validation
    if validate_header:
        cols = list(base.columns)
        if cols != EXPECTED_HEADER:
            raise ValueError(
                f"CSV header mismatch.\nExpected: {EXPECTED_HEADER}\nFound:    {cols}"
            )

    # Ensure path column exists
    if "path" not in base.columns:
        raise ValueError("Input CSV must contain a 'path' column of absolute WAV file paths.")

    # Convert WAVs to mono/16kHz
    converted_paths = convert_to_mono_16khz(
        paths=[Path(p) for p in base["path"].astype(str).tolist()],
        target_dir=conversion_dir,
        ffmpeg_exe=ffmpeg_exe,
    )

    # Extract features
    merged = base.copy()

    if engine in ("librosa", "both"):
        librosa_df = build_librosa_feature_matrix(converted_paths)
        merged = pd.concat(
            [merged.reset_index(drop=True), librosa_df.reset_index(drop=True)], axis=1
        )

    if engine in ("opensmile", "both"):
        opensmile_df = build_opensmile_feature_matrix(converted_paths)
        if engine == "both":  # prefix to avoid column name collisions
            opensmile_df = opensmile_df.add_prefix("smile_")
        merged = pd.concat(
            [merged.reset_index(drop=True), opensmile_df.reset_index(drop=True)], axis=1
        )

    # Step 3 – Write output
    merged.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Done. Wrote {out_csv.resolve()}")

    # Step 4 – Return for programmatic use
    return merged

###############################################################################
# Parser
###############################################################################

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract audio features (Librosa/OpenSMILE) and merge into a CSV."
    )
    p.add_argument("--in_file", type=Path, required=True, help="Input merged CSV path.")
    p.add_argument("--out_file", type=Path, required=True, help="Output CSV path.")
    p.add_argument(
        "--engine",
        type=str,
        choices=["librosa", "opensmile", "both"],
        default="librosa",
        help="Feature extraction engine (default: librosa).",
    )
    p.add_argument(
        "--conversion_dir",
        type=str,
        default="converted_16khz",
        help="Directory to write converted mono/16kHz WAVs.",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip CSV header validation.",
    )
    p.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(rf"C:\ffmpeg\bin\ffmpeg.exe"),
        help="Path to ffmpeg executable (or keep default if on PATH).",
    )
    return p.parse_args()

###############################################################################
# Main
###############################################################################

def main() -> None:
    args = _parse_args()
    merged = extract_and_merge_from_csv(
        in_csv=args.in_file,
        out_csv=args.out_file,
        engine=args.engine,
        validate_header=not args.no_validate,
        conversion_dir=args.conversion_dir,
        ffmpeg_exe=args.ffmpeg,
    )
    print(f"Done. Wrote {args.out_file} (rows: {len(merged)})")

if __name__ == "__main__":
    main()
