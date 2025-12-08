"""
scripts/download_fomc_datasets/download_audio.py
=================================================
Download utilities for the **FOMC press-conference audio dataset**
hosted on Hugging Face: FedSentimentLab/Fed_audio_text_video.

This module provides two core functions:

    • get_audio_files_by_folder(hf_token)
        Queries the dataset via HuggingFace’s virtual filesystem (HfFileSystem)
        and returns a mapping:
            { folder_name : [relative_paths...] }
        where relative paths preserve the original structure
        (e.g., 'audio_files_split/20110427/FILE.wav').

    • download_audio_files_hf(hf_token, files_by_folder, local_root, download_one=False)
        Downloads all (or optionally only one folder of) audio files using
        hf_hub_download. Files are stored under `local_root` while preserving
        their internal relative paths.

Notes
-----
- This module *only* handles listing and downloading.
- Authentication is required: the user must provide a valid `hf_token`
  (typically loaded from `.env` or environment variables).
- Folder structure from HuggingFace is preserved exactly in the local directory.
"""

from typing import Dict, List
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = "FedSentimentLab/Fed_audio_text_video"


def get_audio_files_by_folder(hf_token: str) -> Dict[str, List[str]]:
    audio_files_per_folder: Dict[str, List[str]] = {}

    fs = HfFileSystem(token=hf_token)

    base_dataset_path = "datasets/FedSentimentLab/Fed_audio_text_video/"
    audio_split_path = base_dataset_path + "audio_files_split/"

    folders = [
        item["name"]
        for item in fs.ls(audio_split_path, detail=True)
        if item["type"] == "directory"
    ]

    for folder_path in folders:
        folder_contents = fs.ls(folder_path, detail=True)
        audio_files = [
            item["name"].replace(base_dataset_path, "")
            for item in folder_contents
            if item["type"] == "file"
        ]
        folder_name = folder_path.replace(audio_split_path, "")
        audio_files_per_folder[folder_name] = audio_files

    return audio_files_per_folder

def collect_existing_local_files(local_root: Path) -> Dict[str, Path]:
    """
    Scan the local_root directory and return a mapping:
        {relative_path: absolute_path}

    Assumes the directory structure mirrors the HF dataset:
        local_root/
            audio_files_split/...
    """
    local_root = local_root.resolve()
    existing: Dict[str, Path] = {}

    if not local_root.exists():
        return existing  # nothing to collect

    for path in local_root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_root).as_posix()
            existing[rel] = path.resolve()
    
    return existing

def download_audio_files_hf(
    hf_token: str,
    files_by_folder: Dict[str, List[str]],
    local_root: Path,
    download_one: bool = False,
) -> Dict[str, Path]:
    """
    Download audio files from the Hugging Face dataset into a specified local folder.
    Skips files that already exist locally.
    """
    local_root = local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    local_paths: Dict[str, Path] = collect_existing_local_files(local_root)

    if download_one:
        first_key = list(files_by_folder.keys())[0]
        folders_to_iterate = {first_key: files_by_folder[first_key]}
    else:
        folders_to_iterate = files_by_folder

    for folder_name, file_list in folders_to_iterate.items():
        for rel_path in file_list:

            # Check if already present
            if rel_path in local_paths.keys():
                print(f"Already exists locally: {rel_path}")
                continue
            try:
                local_path_str = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=rel_path,
                    token=hf_token,
                    local_dir=str(local_root),
                    local_dir_use_symlinks=False,
                )
                local_path = Path(local_path_str).resolve()
                local_paths[rel_path] = local_path
                print(f"Downloaded {rel_path} -> {local_path}")

            except Exception as e:
                print(f"Failed to download {rel_path}: {e}")

    return local_paths
