"""
scripts/training_datasets_parsing/TESS.py
=================================================
TESS filename/directory parser and flexible filtering utilities.

This module provides strongly typed data models for the Toronto Emotional Speech
Set (TESS), including tools to scan a directory tree, parse speaker/emotion
labels from folder names, normalize emotion identifiers, and select files via a
clean filtering API.

Overview
--------
- TESS labels emotion *and* speaker identity exclusively via the **parent
  directory name** of each WAV file:
      e.g. "OAF_happy", "YAF_sad", "OAF_Pleasant-Surprise"
- Parsed metadata is represented through immutable dataclasses: `TessId` and
  `TessFile`.
- Emotion queries accept integers or names (case/format-insensitive), all
  normalized through `normalize_tess_emotion[_set]`.
- Filtering supports exact matches, set membership, and mixed-type emotion
  lists.

Key data models
---------------
TessId
    Parsed identity for a TESS directory.
    Fields:
        • speaker: "OAF" (older adult female) or "YAF" (younger adult female)
        • emotion_id: universal emotion ID (via normalize_tess_emotion)

    Derived properties:
        • age_group      -> "older_adult_female" / "younger_adult_female"
        • emotion_name   -> canonical name via UNIVERSAL_EMOTION_ID_TO_NAME

    Construction:
        TessId.from_dirname("OAF_happy")
        TessId.from_dirname("YAF_Pleasant-Surprise")

    Validation:
        - Directory must contain a single underscore separating speaker + label.
        - Speaker must be one of {"OAF", "YAF"}.

TessFile
    Couples:
        • path: Path  (full file location)
        • eid:  TessId (parsed metadata)

    Construction:
        TessFile.from_path(path)   # parent directory defines label

    Filtering (matches):
        Supported keys:
            - speaker="OAF"/"YAF"
            - speaker_in={"OAF","YAF"}
            - emotion=<int or str>           e.g. 3, "happy", "fearful"
            - emotion_in={mixed values}      e.g. {1, "sad", "pleasant-surprise"}
        All emotion filters pass through normalizers.

Filtering logic
---------------
TessFile.matches(**criteria) supports:

• Exact match
      speaker="YAF"
      emotion=3
      emotion="Happy"
      emotion="fearful"

• Set membership
      speaker_in={"OAF","YAF"}
      emotion_in=[1, "happy", "pleasant-surprise", "fear"]

• Mixed types (ints/strings) are allowed and normalized:
      emotion_in=[3, "angry", "Fear"]

• Unknown keys raise ValueError for safety.

Scanning & selection
--------------------
scan_tess(root: Path, pattern: str = "*.wav") -> list[TessFile]
    Recursively discovers WAV files under `root`, parsing each into a TessFile.
    Directory names that do not match the expected "<OAF|YAF>_<emotion>" pattern
    are skipped safely.

select(files, **criteria) -> list[TessFile]
    Functional wrapper applying TessFile.matches() to each element.

main(...)
    Convenience wrapper combining scanning + filtering. Supports:
      - speaker="OAF" / speaker=["OAF","YAF"]
      - emotion=<id|name>
      - emotion_in=<iterable of ids/names>

Parameters (main)
-----------------
root : Path
    Top-level directory containing TESS subfolders.
speaker : str | Iterable[str] | None
    One speaker or multiple speakers ("OAF", "YAF").
emotion : int | str | None
    Single emotion (id or textual alias).
emotion_in : Iterable[int|str] | None
    List/set of mixed ids and names (normalized internally).

Returns
-------
list[TessFile]
    Files matching the filtering criteria.

Examples
--------
>>> root = Path("/datasets/TESS")
>>> # All happy-like utterances from YAF
>>> happy_like = main(root,
...     speaker="YAF",
...     emotion_in=["happy", "pleasant-surprise"]
... )

>>> # Mixed integer–string emotion selection
>>> subset = main(root, emotion_in=[3, "fearful", "sad"])

Notes
-----
- Directory parsing uses dirname.split("_", 1) to keep emotion labels intact
  when they contain hyphens or multiple words (e.g. "pleasant-surprise").
- normalize_tess_emotion handles case/alias mapping ("Fear", "fearful", etc.).
- Dataclasses are frozen=True for immutability and reproducibility.
- scan_tess() is robust by design: invalid folders do not cause crashes.

Output example
--------------
TessFile(
    path=.../OAF_happy/OAF_730_happy.wav,
    eid=TessId(
        speaker="OAF",
        emotion_id=3,
    )
)

All files in this example share:
    speaker="OAF"
    emotion_id=3  -> "happy"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Union

try:
    from utils import (
        normalize_tess_emotion,
        normalize_tess_emotion_set,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )
except ImportError:
    from scripts.training_datasets_parsing.utils import (
        normalize_tess_emotion,
        normalize_tess_emotion_set,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )


###############################################################################
# Data models
###############################################################################

@dataclass(frozen=True)
class TessId:
    """
    Parsed identity for a TESS file inferred from its parent directory name.

    Expected directory names (examples):
      - "OAF_happy", "YAF_sad", "OAF_Pleasant-Surprise", "YAF_FEARFUL", ...

    Fields:
      - speaker: "OAF" or "YAF"
      - emotion_id: universal emotion ID (via utils normalize_tess_emotion)
    """
    speaker: str          # "OAF" or "YAF"
    emotion_id: int       # universal ID (1..9 per utils)

    @property
    def age_group(self) -> str:
        return "younger_adult_female" if self.speaker == "YAF" else "older_adult_female"

    @property
    def emotion_name(self) -> str:
        return UNIVERSAL_EMOTION_ID_TO_NAME[self.emotion_id]

    @classmethod
    def from_dirname(cls, dirname: str) -> "TessId":
        # Expect "<OAF|YAF>_<emotion string>"
        parts = dirname.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid TESS directory name: {dirname}")

        speaker, emo_raw = parts[0].upper(), parts[1]
        if speaker not in {"OAF", "YAF"}:
            raise ValueError(f"Unknown speaker code in {dirname}")

        emo_id = normalize_tess_emotion(emo_raw)
        return cls(speaker=speaker, emotion_id=emo_id)


@dataclass(frozen=True)
class TessFile:
    """
    Couples a Path with its parsed TESS identity.
    """
    path: Path
    eid: TessId

    @classmethod
    def from_path(cls, path: Path) -> "TessFile":
        # Parent directory holds the label in TESS
        eid = TessId.from_dirname(path.parent.name)
        return cls(path=path, eid=eid)

    def matches(self, **criteria: Any) -> bool:
        """
        Supported keys:
          - speaker="OAF"/"YAF"            or speaker_in={"OAF","YAF"}
          - emotion=<int|str>              (e.g. 3, "happy", "fearful")
          - emotion_in={mixed values}      (e.g. {3,"sad","pleasant-surprise"})

        emotion / emotion_in are normalized through utils normalize_tess_emotion(_set).
        """
        for key, val in criteria.items():
            if key == "speaker":
                if self.eid.speaker != str(val).upper():
                    return False

            elif key == "emotion":
                want_id = normalize_tess_emotion(val)
                if self.eid.emotion_id != want_id:
                    return False

            elif key.endswith("_in"):
                field = key[:-3]
                if field == "speaker":
                    allowed = {str(s).upper() for s in val}
                    if self.eid.speaker not in allowed:
                        return False
                elif field == "emotion":
                    allowed_ids = normalize_tess_emotion_set(val)
                    if self.eid.emotion_id not in allowed_ids:
                        return False
                else:
                    raise ValueError(f"Unknown filter key: {key}")

            else:
                raise ValueError(f"Unknown filter key: {key}")

        return True


###############################################################################
# Scanning & selection
###############################################################################

def scan_tess(root: Path, pattern: str = "*.wav") -> List[TessFile]:
    """
    Recursively collect TESS WAV files under `root` and parse them.
    Assumes label is encoded in the immediate parent directory name.
    """
    files: List[TessFile] = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            files.append(TessFile.from_path(p))
        except Exception:
            # Skip anything that doesn't follow the expected layout
            continue
    return files


def select(files: Iterable[TessFile], **criteria: Any) -> List[TessFile]:
    """Filter by the same keys as TessFile.matches()."""
    return [f for f in files if f.matches(**criteria)]


###############################################################################
# Main
###############################################################################

def main(
    root: Path,
    speaker: Optional[Union[str, Iterable[str]]] = None,
    emotion: Optional[Union[int, str]] = None,
    emotion_in: Optional[Iterable[Union[int, str]]] = None,
) -> List[TessFile]:
    """
    Scan TESS at `root` and return files matching the provided criteria.

    Parameters
    ----------
    speaker : "OAF" | "YAF" | Iterable[str] | None
        Single speaker code or multiple via iterable.
    emotion : int | str | None
        Single emotion (universal id or name/alias; normalization via utils).
    emotion_in : Iterable[int|str] | None
        Set/list of emotions (mixed ints/strings allowed; normalization via utils).
    """
    all_files = scan_tess(root)
    print(f"Total parsed files: {len(all_files)}")

    criteria: Dict[str, Any] = {}

    if speaker is not None:
        if isinstance(speaker, str):
            criteria["speaker"] = speaker
        else:
            criteria["speaker_in"] = set(speaker)

    if emotion is not None:
        criteria["emotion"] = emotion              

    if emotion_in is not None:
        criteria["emotion_in"] = list(emotion_in)

    selected = select(all_files, **criteria)
    print(f"Selected files: {len(selected)}")
    for f in selected[:10]:
        try:
            print(" -", f.path.relative_to(root))
        except ValueError:
            print(" -", f.path)
    return selected


if __name__ == "__main__":
    # Example local test 
    root = Path(__file__).resolve().parents[2] / "dataset_training" / "TESS"
    selected = main(root, speaker=["OAF", "YAF"], emotion_in=["happy", "fear", 4, "pleasant-surprise"])
