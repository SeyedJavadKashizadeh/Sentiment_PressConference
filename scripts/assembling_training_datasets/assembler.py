"""
scripts/assembling_training_datasets/assembler.py
=================================================
Utilities for assembling **training audio datasets** (EmoDB, RAVDESS, TESS)
into a single merged manifest that can later be used for feature extraction.

This module delegates all dataset-specific parsing to the corresponding
dataset parsers in `scripts.training_datasets_parsing` and provides:

    • normalize_emotion_inputs(...)
        Convert mixed emotion specifications (IDs or names) into canonical
        universal emotion IDs.

    • assemble_all_datasets(dataset_root, gender, emotions)
        Run all three dataset parsers (EmoDB, RAVDESS, TESS) and return a
        dictionary mapping dataset name -> list of parsed file objects.

    • save_manifest(results, output_folder)
        Write the merged CSV manifest with columns:
            dataset, path, emotion_name, emotion_code

Assumptions
-----------
- The training datasets already exist locally under `dataset_root` in the
  following structure:
        dataset_root/
            EmoDB/
            RAVDESS/
            TESS/

- Each dataset has its own parser implementing a `main()` function that
  returns normalized audio-file metadata (path + emotion info).

- Gender filtering is supported for EmoDB and RAVDESS. TESS uses only female
  speakers; the `speaker` argument is handled internally.

Use case
--------
This module is typically used by the orchestration script
`helpers/run_extract_training_features.py` to build the merged `merged.csv`
before extracting acoustic features.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union, Any

try:
    from scripts.training_datasets_parsing.EMODB import main as emodb_main
    from scripts.training_datasets_parsing.RAVDESS import main as ravdess_main
    from scripts.training_datasets_parsing.TESS import main as tess_main

    from scripts.training_datasets_parsing.utils import (
        normalize_universal_emotion,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )

except Exception:
    from training_datasets_parsing.EMODB import main as emodb_main  # type: ignore
    from training_datasets_parsing.RAVDESS import main as ravdess_main  # type: ignore
    from training_datasets_parsing.TESS import main as tess_main  # type: ignore

    from training_datasets_parsing.utils import (  # type: ignore
        normalize_universal_emotion,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )

###############################################################################
# Constants
###############################################################################

VALID_GENDERS = {"male", "female"}

###############################################################################
# Data models
###############################################################################

@dataclass(frozen=True)
class DatasetSelection:
    """A small container to carry per-dataset selections (paths only)."""
    dataset: str
    paths: List[Path]

###############################################################################
# Parsers
###############################################################################

def normalize_emotion_inputs(vals: Sequence[Union[int, str]]) -> List[int]:
    """
    Accept a mixed list of ints/strings and produce canonical universal IDs.

    Examples
    --------
    normalize_emotion_inputs([3, "happy", "fear", 4, "pleasant-surprise"])
    -> [3, 6, 4, 8] (order preserved)
    """
    out: List[int] = []
    for v in vals:
        out.append(normalize_universal_emotion(v))
    return out

###############################################################################
# Helpers
###############################################################################

def save_manifest(results: dict[str, list[Any]], output_folder: Path) -> Path:
    """
    Create CSV with columns: dataset, path, emotion_name, emotion_code.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    out_csv = output_folder / "merged.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:           
        writer = csv.writer(f)
        writer.writerow(["dataset", "path", "emotion_name", "emotion_code"])
        for ds, files in results.items():
            for file in files:
                path = file.path
                emotion_id = file.eid.emotion_id
                writer.writerow([ds, path, UNIVERSAL_EMOTION_ID_TO_NAME.get(emotion_id), emotion_id])
    
    return out_csv

###############################################################################
# Core function
###############################################################################

def assemble_all_datasets(
    dataset_root: Path,
    gender: str | None,
    emotions: Iterable[Union[int, str]],
) -> Dict[str, List[Path]]:
    """
    Orchestrates the three dataset selectors and returns a dict:
        {"EmoDB": [...files], "RAVDESS": [...], "TESS": [...]}

    Parameters
    ----------
    dataset_root : Path
        Folder that contains 'EmoDB', 'RAVDESS', and 'TESS' subfolders.
    gender : Optional[str]
        "male" / "female" or None.
    emotions : Iterable[Union[int,str]]
    Notes
    -----
    - Each dataset `main()` has its own interface:
        * EmoDB: main(root, gender=?, emotion_in=?)
        * RAVDESS: main(root, gender=?, emotion_in=?)
        * TESS: main(root, speaker=?, emotion_in=?)  # TESS uses "speaker" = OAF/YAF.
    """
    root_emodb   = dataset_root / "EmoDB"
    root_ravdess = dataset_root / "RAVDESS"
    root_tess    = dataset_root / "TESS"

    # EMODB
    emodb_files = emodb_main(
        root=root_emodb,
        gender=gender,
        emotion_in=emotions,
    )

    # RAVDESS
    ravdess_files = ravdess_main(
        root=root_ravdess,
        gender=gender,
        emotion_in=emotions,
    )

    # TESS
    speaker = None
    if gender is not None:
        speaker = {"male": None, "female": None}.get(gender, None)

    tess_files = tess_main(
        root=root_tess,
        speaker=speaker,
        emotion= None,
        emotion_in=emotions,
    )

    results: Dict[str, List[Any]] = {
        "emodb" : emodb_files,
        "ravdess" : ravdess_files,
        "tess" : tess_files
    }

    return results


