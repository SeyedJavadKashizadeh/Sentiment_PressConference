"""
scripts/download_training_datasets/download_audio.py
=================================================

Download and listing utilities for **training emotion datasets** from Kaggle.

This module centralises:
    • Download of emotional-speech corpora (RAVDESS, TESS, EmoDB) via KaggleHub,
      into a local root directory (e.g. <repo>/data_training/raw_data).

    • Listing of local audio files per dataset once downloaded, searching
      recursively for common audio extensions (*.wav, *.flac, *.mp3).

Main functions
--------------
download_audio_files_kaggle(local_root, download_one=False)
    Ensure one or all datasets in DATASET_MAP are present under `local_root`,
    downloading and copying them from the Kaggle cache if needed.

get_audio_files_by_dataset(local_root)
    Return a mapping {dataset_name: [Path(...)]} of all audio files found
    under each dataset subfolder in `local_root`.
"""

from typing import Dict, List
from pathlib import Path
import shutil

import kagglehub

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data_training" / "raw_data"

# Mapping from local dataset name -> Kaggle dataset identifier
DATASET_MAP: Dict[str, str] = {
    "RAVDESS": "uwrfkaggler/ravdess-emotional-speech-audio",
    "TESS": "ejlok1/toronto-emotional-speech-set-tess",
    "EmoDB": "piyushagni5/berlin-database-of-emotional-speech-emodb",
}

################
# Helper for Tess
################
def find_real_tess_root(cached_dir: Path) -> Path:
    """
    Detects and returns the actual root directory of TESS audio folders,
    avoiding duplicated wrapper folders from the Kaggle archive.
    """
    current = cached_dir

    seen = set()
    while True:
        subdirs = [d for d in current.iterdir() if d.is_dir()]

        # stop if multiple subdirectories → likely the real audio level
        if len(subdirs) != 1:
            return current

        next_dir = subdirs[0]

        # detect infinite nesting of same-name folders
        if next_dir.name in seen:
            break

        seen.add(next_dir.name)
        current = next_dir

    return current

def get_audio_files_by_dataset(local_root: Path) -> Dict[str, List[Path]]:
    """
    List audio files per dataset under a given local root directory.

    This assumes that each dataset has already been downloaded into a
    subfolder of `local_root` named after the keys of DATASET_MAP, i.e.:

        <local_root>/
            RAVDESS/
            TESS/
            EmoDB/

    For each existing dataset subfolder, this function recursively searches
    for common audio file extensions (*.wav, *.flac, *.mp3) and returns
    a mapping from dataset name to the list of audio file paths.

    Parameters
    ----------
    local_root : Path
        Root directory under which training datasets are stored.

    Returns
    -------
    dict
        Mapping {dataset_name: [audio_file_paths]} where each audio_file_path
        is a Path object pointing to a local audio file.
    """
    local_root = local_root.resolve()
    audio_files_per_dataset: Dict[str, List[Path]] = {}

    for dataset_name in DATASET_MAP.keys():
        dataset_dir = local_root / dataset_name
        if not dataset_dir.exists():
            # Dataset not downloaded yet; skip it
            continue

        # Collect audio files (extend pattern list if needed)
        files: List[Path] = []
        for pattern in ("*.wav", "*.flac", "*.mp3"):
            files.extend(dataset_dir.rglob(pattern))

        audio_files_per_dataset[dataset_name] = files

    return audio_files_per_dataset


def download_audio_files_kaggle(
    local_root: Path,
    download_one: bool = False,
) -> Dict[str, Path]:
    """
    Download emotional-speech training datasets from Kaggle into a local folder.

    Parameters
    ----------
    local_root : Path
        Local directory under which the dataset folders will be stored.
        Each dataset will be copied into:
            <local_root>/<dataset_name>/
        where dataset_name is one of DATASET_MAP keys
        (e.g., 'RAVDESS', 'TESS', 'EmoDB').
    download_one : bool, optional
        If True, only the first dataset in DATASET_MAP is downloaded.
        If False, all datasets in DATASET_MAP are downloaded. Default is False.

    Returns
    -------
    dict
        Mapping {dataset_name: local_dataset_dir} where local_dataset_dir
        is the Path to the root folder of that dataset under local_root.

    Notes
    -----
    - If a dataset directory already exists under local_root, it is left
      untouched and considered "already downloaded".
    - Downloads use `kagglehub.dataset_download(kaggle_id)` to retrieve the
      cached directory, which is then copied into local_root/<dataset_name>.
    """
    local_root = local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    
    local_paths: Dict[str, Path] = {}

    # Restrict to first dataset if requested
    items = list(DATASET_MAP.items())
    if download_one:
        items = items[:1]

    for dataset_name, kaggle_id in items:
        if local_root.name == dataset_name:
            target_dir = local_root
        else:
            target_dir = local_root / dataset_name
        
        print(f"[{dataset_name}] local_root = {local_root}")
        print(f"[{dataset_name}] target_dir = {target_dir}")

        # If dataset already exists, skip download
        if target_dir.exists():
            print(f"Dataset '{dataset_name}' already present at {target_dir}")
            local_paths[dataset_name] = target_dir.resolve()
            continue

        try:
            # Download (or reuse cached) Kaggle dataset directory
            cached_path = kagglehub.dataset_download(kaggle_id)
            cached_dir = Path(cached_path).resolve()
            print(f"Downloaded '{dataset_name}' to Kaggle cache: {cached_dir}")

            # Avoid downloading twice TESS datafile because the publisher put twice the files in the folder
            if dataset_name == "TESS":
                src_dir = find_real_tess_root(cached_dir)
            else:
                src_dir = cached_dir
            
            # Copy from Kaggle cache into our desired location
            shutil.copytree(src_dir, target_dir, dirs_exist_ok=True)
            print(f"Copied '{dataset_name}' dataset to: {target_dir}")

            if dataset_name == "TESS":
                nested_dup = target_dir / "TESS Toronto emotional speech set data"
                if nested_dup.exists():
                    print(f"Removing nested duplicate TESS dir: {nested_dup}")
                    shutil.rmtree(nested_dup)

            local_paths[dataset_name] = target_dir.resolve()

        except Exception as e:
            print(f"Failed to download dataset '{dataset_name}' ({kaggle_id}): {e}")

    return local_paths
