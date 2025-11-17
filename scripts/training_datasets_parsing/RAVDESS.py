"""
scripts/training_datasets_parsing/RAVDESS.py
=================================================
RAVDESS filename parser and flexible filter utilities.

This module provides strongly typed data models for the RAVDESS dataset and a
small engine to scan a directory tree, parse filenames into structured
metadata, and select files via expressive criteria (including human-friendly
emotion names).

Overview
--------
- Metadata is extracted **only** from the filename stem
  (e.g. "03-01-06-01-02-01-12").
- Parsed fields are validated (ranges, neutral intensity rule) and exposed via
  immutable dataclasses: `RavdessId` and `RavdessFile`.
- Emotion filters accept either integers or names; all are normalized to a
  universal id space through `normalize_ravdess_emotion[_set]`.

Key data models
---------------
RavdessId
    Structured view of "MM-VC-EM-IN-ST-RP-AC".
    Fields (ints): modality(1..3), vocal_channel(1..2), emotion_code(1..8),
    intensity(1..2), statement(1..2), repetition(1..2), actor(1..24).
    Derived:
        • gender: "male" if actor is odd, else "female".
        • emotion_name, modality_name, vocal_channel_name, intensity_name,
          statement_text via the dictionaries imported from `utils`.

    Construction:
        `RavdessId.from_stem("03-01-06-01-02-01-12")`

    Serialization:
        `eid.to_stem()  -> "03-01-06-01-02-01-12"`

    Validation rules:
        • All numeric parts must be in their documented ranges.
        • Neutral (emotion_code==1) **cannot** have strong intensity (2).

RavdessFile
    Couples a `path: Path` with its parsed `eid: RavdessId`.
    Use `RavdessFile.from_path(path)` to construct.
    `.matches(**criteria)` applies flexible filtering (see below).

Filtering API
-------------
`RavdessFile.matches(**criteria)` supports:

Exact match
    modality=3, vocal_channel=1, intensity=2, statement=1, repetition=2,
    actor=17, emotion=3 (int) or emotion="sad" (name).

Set membership
    modality_in={1,3}, actor_in=[1,3,5], repetition_in={1,2},
    emotion_in=[1, "happy", "fear"]  # ints and names mixed.

Derived attribute
    gender="male" or "female".

Invalid filter keys raise `ValueError`. For emotions, names/ints are normalized
via `normalize_ravdess_emotion[_set]` before comparison.

Scanning & selection
--------------------
scan_ravdess(root: Path, pattern: str = "*.wav") -> list[RavdessFile]
    Recursively discovers WAV files under `root`, skipping the official folder
    "audio_speech_actors_01-24" if present. Non-conforming filenames are
    ignored.

select(files: Iterable[RavdessFile], **criteria) -> list[RavdessFile]
    Returns files satisfying `.matches(**criteria)`.

main(...)
    Convenience wrapper that scans then filters. Supports all criteria listed
    above, including mixed-type emotion filters. Prints a short summary and
    returns the selected files.

Parameters (main)
-----------------
root : Path
    Root directory to scan.
gender : {"male","female"}, optional
modality, vocal_channel, intensity, statement, repetition, actor : int, optional
*_in : Iterable[int], optional
    Set-membership versions of the above (e.g., modality_in, actor_in, ...).
emotion : int | str, optional
    Either a universal id (1..8 for RAVDESS) or a name like "happy", "sad".
emotion_in : Iterable[int | str], optional
    Mixed list of ids/names.

Returns
-------
list[RavdessFile]
    The files whose parsed metadata match the provided criteria.

Examples
--------
>>> root = Path("/data/RAVDESS")
>>> # All female SAD utterances (emotion by name)
>>> subset = main(root, gender="female", emotion="sad")
>>> # Mixed integer–string emotion filter
>>> subset = main(root, emotion_in=[1, "happy", "fear"])
>>> # Audio-only male speech with strong intensity
>>> subset = main(root, gender="male", modality=3, vocal_channel=1, intensity=2)

Notes
-----
- Mappings (EMOTION_ID_TO_NAME, MODALITY, VOCAL_CHANNEL, EMOTION_INTENSITY,
  STATEMENT) and normalizers are imported from `utils`.
- Dataclasses are `frozen=True` for immutability and hash safety.
- Directory layout is irrelevant; only filename stems are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Union

try:
    from utils import (
        UNIVERSAL_EMOTION_ID_TO_NAME as EMOTION_ID_TO_NAME,   
        RAVDESS_STATEMENT as STATEMENT,                      
        RAVDESS_MODALITY as MODALITY,                         
        RAVDESS_VOCAL_CHANNEL as VOCAL_CHANNEL,               
        RAVDESS_EMOTION_INTENSITY as EMOTION_INTENSITY,       
        normalize_ravdess_emotion,                            
        normalize_ravdess_emotion_set,                        
    )
except ImportError:
    from scripts.training_datasets_parsing.utils import (
        UNIVERSAL_EMOTION_ID_TO_NAME as EMOTION_ID_TO_NAME,   
        RAVDESS_STATEMENT as STATEMENT,                       
        RAVDESS_MODALITY as MODALITY,                         
        RAVDESS_VOCAL_CHANNEL as VOCAL_CHANNEL,               
        RAVDESS_EMOTION_INTENSITY as EMOTION_INTENSITY,       
        normalize_ravdess_emotion,                            
        normalize_ravdess_emotion_set,                        
    )
###############################################################################
# Data models
###############################################################################

@dataclass(frozen=True)
class RavdessId:
    """
    Parsed RAVDESS identifier from a filename stem 'MM-VC-EM-IN-ST-RP-AC'.

    Fields store canonical integer codes as per the official RAVDESS spec.
    Emotion is kept as a *universal* emotion id (1..8) for cross-dataset
    consistency (via normalize_ravdess_emotion).
    """
    modality: int         # 01..03
    vocal_channel: int    # 01..02
    emotion_id: int     # universal id after normalization/collapse
    intensity: int        # 01..02  (no '02' when emotion==1 "neutral")
    statement: int        # 01..02
    repetition: int       # 01..02
    actor: int            # 01..24

    @property
    def gender(self) -> str:
        # Odd actors are male; even actors are female in RAVDESS
        return "male" if self.actor % 2 == 1 else "female"

    @property
    def emotion_name(self) -> str:
        return EMOTION_ID_TO_NAME.get(self.emotion_id, f"unknown({self.emotion_id})")

    @property
    def modality_name(self) -> str:
        return MODALITY.get(self.modality, f"unknown({self.modality})")

    @property
    def vocal_channel_name(self) -> str:
        return VOCAL_CHANNEL.get(self.vocal_channel, f"unknown({self.vocal_channel})")

    @property
    def intensity_name(self) -> str:
        return EMOTION_INTENSITY.get(self.intensity, f"unknown({self.intensity})")

    @property
    def statement_text(self) -> str:
        return STATEMENT.get(self.statement, f"unknown({self.statement})")

    @classmethod
    def from_stem(cls, stem: str) -> "RavdessId":
        """
        Parse '03-01-06-01-02-01-12' into fields and validate ranges.
        Emotion is normalized to the universal id space (1..8 for RAVDESS).
        """
        parts = stem.split("-")
        if len(parts) != 7 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Invalid RAVDESS stem: {stem}")

        m, vc, emo_raw, inten, stmt, rep, act = (int(p) for p in parts)

        # Validate fixed ranges
        if not (1 <= m <= 3):      raise ValueError(f"modality out of range in {stem}")
        if not (1 <= vc <= 2):     raise ValueError(f"vocal_channel out of range in {stem}")
        if not (1 <= inten <= 2):  raise ValueError(f"intensity out of range in {stem}")
        if not (1 <= stmt <= 2):   raise ValueError(f"statement out of range in {stem}")
        if not (1 <= rep <= 2):    raise ValueError(f"repetition out of range in {stem}")
        if not (1 <= act <= 24):   raise ValueError(f"actor out of range in {stem}")

        # RAVDESS uses 1..8 for emotions; normalize through the universal mapper
        emo = normalize_ravdess_emotion(emo_raw)

        # Neutral has no 'strong' intensity
        if emo_raw == 1 and inten != 1:
             raise ValueError(f"neutral emotion cannot have strong intensity in {stem}")
        
        return cls(m, vc, emo, inten, stmt, rep, act)

    def to_stem(self) -> str:
        # Serialize back to the standard filename stem (using universal emotion id)
        return (
            f"{self.modality:02d}-{self.vocal_channel:02d}-{self.emotion_id:02d}-"
            f"{self.intensity:02d}-{self.statement:02d}-{self.repetition:02d}-{self.actor:02d}"
        )


@dataclass(frozen=True)
class RavdessFile:
    """
    Couple a filesystem Path with its parsed RavdessId.
    """
    path: Path
    eid: RavdessId

    @classmethod
    def from_path(cls, path: Path) -> "RavdessFile":
        return cls(path=path, eid=RavdessId.from_stem(path.stem))

    def matches(self, **criteria: Any) -> bool:
        """
        Flexible filtering with emotion normalization:
          - gender="male"/"female"
          - modality / modality_in
          - vocal_channel / vocal_channel_in
          - emotion or emotion_code (int or name); emotion_in can mix ints/strings
          - intensity / intensity_in
          - statement / statement_in
          - repetition / repetition_in
          - actor / actor_in
        """
        for key, value in criteria.items():

            # Derived attribute
            if key == "gender":
                if self.eid.gender != value:
                    return False
                continue

            # Emotion (accepts int or string label/alias)
            if key in {"emotion", "emotion_code"}:
                want = normalize_ravdess_emotion(value)
                if self.eid.emotion_id != want:
                    return False
                continue

            if key in {"emotion_in", "emotion_code_in"}:
                allowed = normalize_ravdess_emotion_set(value)
                if self.eid.emotion_id not in allowed:
                    return False
                continue

            # Generic *_in for the remaining integer fields
            if key.endswith("_in"):
                field = key[:-3]
                if field == "emotion":   # alias to stored attribute name
                    field = "emotion_code"
                if not hasattr(self.eid, field):
                    raise ValueError(f"Unknown filter key: {key}")
                if getattr(self.eid, field) not in value:
                    return False
                continue

            # Exact match (with emotion alias handled above)
            field = "emotion_code" if key == "emotion" else key
            if not hasattr(self.eid, field):
                raise ValueError(f"Unknown filter key: {key}")
            if getattr(self.eid, field) != value:
                return False

        return True

###############################################################################
# Scanning & selection
###############################################################################

def scan_ravdess(root: Path, pattern: str = "*.wav") -> List[RavdessFile]:
    """
    Recursively collect WAV files under `root`, skipping the official
    'audio_speech_actors_01-24' folder if present, and parse them.
    """
    files: List[RavdessFile] = []
    skip = "audio_speech_actors_01-24"
    for p in root.rglob(pattern):
        if skip in p.parts:
            continue
        if not p.is_file():
            continue
        try:
            files.append(RavdessFile.from_path(p))
        except Exception:
            # Skip files that don't conform to the naming scheme
            continue
    return files


def select(files: Iterable[RavdessFile], **criteria: Any) -> List[RavdessFile]:
    """Filter a collection of RavdessFile by attribute criteria."""
    return [f for f in files if f.matches(**criteria)]

###############################################################################
# Main
###############################################################################

def main(
    root: Path,
    gender: Optional[str] = None,
    modality: Optional[int] = None,
    modality_in: Optional[Iterable[int]] = None,
    vocal_channel: Optional[int] = None,
    vocal_channel_in: Optional[Iterable[int]] = None,
    emotion: Optional[Union[int, str]] = None,
    emotion_in: Optional[Iterable[Union[int, str]]] = None,
    intensity: Optional[int] = None,
    intensity_in: Optional[Iterable[int]] = None,
    statement: Optional[int] = None,
    statement_in: Optional[Iterable[int]] = None,
    repetition: Optional[int] = None,
    repetition_in: Optional[Iterable[int]] = None,
    actor: Optional[int] = None,
    actor_in: Optional[Iterable[int]] = None,
) -> List[RavdessFile]:
    """
    Scan RAVDESS under `root` and return files matching the given criteria.
    Accepts mixed emotion filters (ints or strings).
    """
    all_files = scan_ravdess(root)
    print(f"Total parsed files: {len(all_files)}")

    criteria: Dict[str, Any] = {}

    if gender is not None: criteria["gender"] = gender

    if modality is not None: criteria["modality"] = modality
    if modality_in is not None: criteria["modality_in"] = set(modality_in)

    if vocal_channel is not None: criteria["vocal_channel"] = vocal_channel
    if vocal_channel_in is not None: criteria["vocal_channel_in"] = set(vocal_channel_in)

    # Emotion: pass raw; normalization happens inside .matches()
    if emotion is not None: criteria["emotion"] = emotion
    if emotion_in is not None: criteria["emotion_in"] = set(emotion_in)

    if intensity is not None: criteria["intensity"] = intensity
    if intensity_in is not None: criteria["intensity_in"] = set(intensity_in)

    if statement is not None: criteria["statement"] = statement
    if statement_in is not None: criteria["statement_in"] = set(statement_in)

    if repetition is not None: criteria["repetition"] = repetition
    if repetition_in is not None: criteria["repetition_in"] = set(repetition_in)

    if actor is not None: criteria["actor"] = actor
    if actor_in is not None: criteria["actor_in"] = set(actor_in)

    selected = select(all_files, **criteria)

    print(f"Selected files: {len(selected)}")
    for f in selected[:5]:
        print(" -", f.path.name)

    return selected


if __name__ == "__main__":
    # Example local test 
    root = Path(__file__).resolve().parents[2] / "data_training/raw_data" / "RAVDESS"
    selected = main(root, gender="female", emotion_in=["happy", "fear", 4])