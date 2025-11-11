"""
scripts/assembling_training_datasets/assembler.py
=================================================
Library code that orchestrates dataset-specific parsers to assemble a unified
list of training audio file paths.

This module imports your dataset parsers:

- scripts.training_datasets_parsing.EMODB.main
- scripts.training_datasets_parsing.RAVDESS.main
- scripts.training_datasets_parsing.TESS.main

and runs them with consistent filters:
    * gender: "male" / "female" (applies to EmoDB & RAVDESS; ignored for TESS)
    * emotions: a list of items (id/name/code) passed through to each parser.
      Each parser is responsible for normalizing emotion inputs.

Return value is a dict:
    {
        "emodb":   List[Path],
        "ravdess": List[Path],
        "tess":    List[Path],
    }
with only the datasets you ask for.

Design notes
------------
- We return bare Paths to keep a common type across datasets (their parsers
  return dataset-specific dataclasses; we convert to .path uniformly).
- TESS does not support a gender filter (the dataset is all female). If
  a gender is requested, we print a one-line info and proceed ignoring it.
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


