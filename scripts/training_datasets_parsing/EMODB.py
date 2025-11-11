"""
scripts/training_datasets_parsing/EMODB.py
=================================================
EmoDB parser, normalizer, scanner, and filter.

OVERVIEW
--------
A strictly validated, strongly typed toolkit for the Berlin Emotional Speech
(EmoDB) dataset. All metadata is parsed **only** from filename stems
(e.g., "16b10Td"), yielding:

- `EmoId`   : immutable record (speaker_code, emotion_id in universal space)
- `EmoFile` : (path, EmoId) pair with a flexible `.matches()` filter

This implementation supports **smart emotion normalization**:
    emotion=4
    emotion="T"          # dataset letter code
    emotion="sad"        # human label
    emotion_in=[4, "W", "fear", "N"]

All accepted forms are mapped to the canonical universal emotion IDs via
`normalize_emodb_emotion(_set)`.

FILENAME FORMAT & REGEX
-----------------------
Expected compact EmoDB stems match:

    ^(?P<speaker>\d{2})([A-Za-z])\d{2}(?P<emotion>[A-Za-z])[A-Za-z]?$

Examples:
    "03a01Fa", "16b10Td", ...

Parsing extracts:
    - speaker_code : "03", "16", ...
    - emotion_code : single letter (e.g., "T", "W", "N"), later normalized

EMOTION SPACE
-------------
Universal IDs available in EmoDB: {1, 3, 4, 5, 6, 7, 9}
Mappings are provided by:
    - EMODB_UNIV_TO_CODE           # {universal_id -> "N"|"F"|...}
    - UNIVERSAL_EMOTION_ID_TO_NAME # {universal_id -> "neutral"|"happy"|...}

DATA MODELS
-----------
EmoId(speaker_code: str, emotion_id: int)
    Properties:
        • gender        -> "male" | "female" | "unknown"
        • emotion_code  -> dataset letter code ("N","F","T","W","A","E","L")
        • emotion_name  -> universal label ("neutral","happy","sad",...)

    Constructor:
        EmoId.from_stem("16b10Td")

EmoFile(path: Path, eid: EmoId)
    Constructor:
        EmoFile.from_path(path)

    Filtering:
        .matches(**criteria)

FILTERING KEYS
--------------
Supported by `EmoFile.matches(**criteria)`:

    # Exact
    gender="female"
    speaker_code="16"
    emotion=4                 # id
    emotion="T"               # letter code
    emotion="sad"             # name

    # Membership (use *_in)
    speaker_code_in={"03","08","16"}
    gender_in={"male","female"}
    emotion_in=[4, "W", "fear", "N"]  # mixed id/code/name

Unknown keys raise ValueError. All emotion inputs are normalized to universal IDs.

SCANNING & SELECTION
--------------------
scan_emodb(root: Path, pattern="*.wav") -> List[EmoFile]
    Recursively finds WAV files under `root`, parses stems, and skips non-conforming
    filenames.

select(files: Iterable[EmoFile], **criteria) -> List[EmoFile]
    Convenience wrapper applying `.matches()` to each file.

MAIN ENTRY POINT
----------------
main(
    root: Path,
    gender: Optional[str] = None,
    speaker_code: Optional[str] = None,
    speaker_code_in: Optional[Iterable[str]] = None,
    emotion: Optional[Any] = None,              # id | letter code | name
    emotion_in: Optional[Iterable[Any]] = None, # mix of the above
) -> List[EmoFile]

Workflow:
    1) scan_emodb(root)
    2) build `criteria` (emotion inputs normalized up-front)
    3) select(...)
    4) print summary & return the filtered list

EXAMPLES
--------
root = Path("/path/to/EmoDB")

# Female anger (letter code)
subset = main(root, gender="female", emotion="W")

# Sad or boredom (names)
subset = main(root, emotion_in=["sad", "boredom"])

# Specific speakers
subset = main(root, speaker_code_in={"03","08"})

GUARANTEES
----------
- Parsing is strict and fails fast on malformed stems.
- Gender is inferred from speaker sets (MALE_SPEAKER_CODES / FEMALE_SPEAKER_CODES).
- Dataclasses are frozen (immutable, hash-safe).
- Filesystem layout is irrelevant; only filename stems are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Any
import re

try:
    from utils import (
        normalize_emodb_emotion,
        normalize_emodb_emotion_set,
        EMODB_UNIV_TO_CODE,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )
except ImportError:
    from scripts.training_datasets_parsing.utils import (
        normalize_emodb_emotion,
        normalize_emodb_emotion_set,
        EMODB_UNIV_TO_CODE,
        UNIVERSAL_EMOTION_ID_TO_NAME,
    )


###############################################################################
# Constants
###############################################################################

MALE_SPEAKER_CODES   = {"03", "10", "11", "12", "15"}
FEMALE_SPEAKER_CODES = {"08", "09", "13", "14", "16"}
ALL_SPEAKER_CODES    = MALE_SPEAKER_CODES | FEMALE_SPEAKER_CODES

# Build a reverse map {code -> universal_id} from utils' EMODB_UNIV_TO_CODE
EMOTION_ID_TO_CODE: Dict[int, str] = dict(EMODB_UNIV_TO_CODE)
EMOTION_CODE_TO_ID: Dict[str, int] = {code: eid for eid, code in EMOTION_ID_TO_CODE.items()}

###############################################################################
# Filename parsing
###############################################################################

_STEM_RE = re.compile(r"^(?P<speaker>\d{2})([A-Za-z])\d{2}(?P<emotion>[A-Za-z])[A-Za-z]?$")

def _parse_stem(stem: str) -> tuple[str, str]:
    """
    Parse an EmoDB filename stem like '16b10Td' -> ('16', 'T')
    """
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Invalid EmoDB filename stem: {stem}")
    speaker = m.group("speaker")
    emo_code = m.group("emotion").upper()
    return speaker, emo_code

###############################################################################
# Data models
###############################################################################

@dataclass(frozen=True)
class EmoId:
    """
    Parsed identity for an EmoDB file, in universal emotion space.
    - speaker_code: e.g. "16"
    - emotion_id:   universal ID (e.g. 4 for 'sad')
    """
    speaker_code: str
    emotion_id: int  # universal emotion ID (1,3,4,5,6,7,9 for EmoDB)

    @property
    def gender(self) -> str:
        if self.speaker_code in MALE_SPEAKER_CODES:
            return "male"
        if self.speaker_code in FEMALE_SPEAKER_CODES:
            return "female"
        return "unknown"

    @property
    def emotion_code(self) -> str:
        # Single-letter dataset code via utils mapping
        return EMOTION_ID_TO_CODE[self.emotion_id]

    @property
    def emotion_name(self) -> str:
        # Human-friendly universal name, e.g. "sad"
        return UNIVERSAL_EMOTION_ID_TO_NAME[self.emotion_id]

    @classmethod
    def from_stem(cls, stem: str) -> "EmoId":
        speaker, emo_code = _parse_stem(stem)

        if speaker not in ALL_SPEAKER_CODES:
            raise ValueError(f"Unknown speaker code '{speaker}' in {stem}")

        if emo_code not in EMOTION_CODE_TO_ID:
            raise ValueError(f"Unknown emotion code '{emo_code}' in {stem}")

        return cls(speaker_code=speaker, emotion_id=EMOTION_CODE_TO_ID[emo_code])


@dataclass(frozen=True)
class EmoFile:
    """
    Couples a file path with its parsed EmoId and provides flexible filtering.
    """
    path: Path
    eid: EmoId

    @classmethod
    def from_path(cls, path: Path) -> "EmoFile":
        return cls(path=path, eid=EmoId.from_stem(path.stem))

    def matches(self, **criteria: Any) -> bool:
        """
        Supported filters:
          - gender="male"/"female"
          - speaker_code="03" or speaker_code_in={...}
          - emotion=<id|code|name>           e.g. 4, "T", "sad"
          - emotion_in={mix of id/code/name} e.g. [4, "W", "fear", "N"]
        """
        for key, value in criteria.items():
            if key == "gender":
                if self.eid.gender != value:
                    return False

            elif key == "speaker_code":
                if self.eid.speaker_code != value:
                    return False

            elif key == "emotion":
                # normalize via utils (accepts id/code/name) → universal ID
                want_id = normalize_emodb_emotion(value)
                if self.eid.emotion_id != want_id:
                    return False

            elif key.endswith("_in"):
                field = key[:-3]
                if field == "speaker_code":
                    if self.eid.speaker_code not in value:
                        return False
                elif field == "gender":
                    if self.eid.gender not in value:
                        return False
                elif field == "emotion":
                    # normalize set via utils → set of universal IDs
                    want_ids = normalize_emodb_emotion_set(value)
                    if self.eid.emotion_id not in want_ids:
                        return False
                else:
                    raise ValueError(f"Unknown filter key: {key}")

            else:
                raise ValueError(f"Unknown filter key: {key}")

        return True

###############################################################################
# Scanning & selection
###############################################################################

def scan_emodb(root: Path, pattern: str = "*.wav") -> List[EmoFile]:
    """
    Recursively collect EmoDB WAV files under `root` and parse them.
    Non-conforming filenames are skipped.
    """
    files: List[EmoFile] = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            files.append(EmoFile.from_path(p))
        except Exception:
            # Skip files that don't match the EmoDB scheme
            continue
    return files


def select(files: Iterable[EmoFile], **criteria: Any) -> List[EmoFile]:
    """Filter a collection of EmoFile by attribute criteria."""
    return [f for f in files if f.matches(**criteria)]

###############################################################################
# Main
###############################################################################

def main(
    root: Path,
    gender: Optional[str] = None,
    speaker_code: Optional[str] = None,
    speaker_code_in: Optional[Iterable[str]] = None,
    emotion: Optional[Any] = None,                
    emotion_in: Optional[Iterable[Any]] = None,
) -> List[EmoFile]:
    """
    Scan EmoDB under `root` and return files matching given criteria.
    Examples:
        emotion=4
        emotion="T"
        emotion="sad"
        emotion_in=[4, "W", "fear", "N"]
    """
    all_files = scan_emodb(root)
    print(f"Total parsed files: {len(all_files)}")

    criteria: Dict[str, Any] = {}
    if gender is not None:
        criteria["gender"] = gender

    if speaker_code is not None:
        criteria["speaker_code"] = speaker_code
    if speaker_code_in is not None:
        criteria["speaker_code_in"] = set(speaker_code_in)

    if emotion is not None:
        # we can normalize here or inside matches(); normalizing now is fine:
        criteria["emotion"] = normalize_emodb_emotion(emotion)

    if emotion_in is not None:
        criteria["emotion_in"] = normalize_emodb_emotion_set(emotion_in)

    selected_files = select(all_files, **criteria)
    print(f"Selected files: {len(selected_files)}")
    for f in selected_files[:5]:
        print(" -", f.path.name)
    return selected_files


if __name__ == "__main__":
    # Example local test 
    root = Path(__file__).resolve().parents[2] / "dataset_training" / "EmoDB"
    _ = main(
        root,
        gender="female",
        emotion_in=[4, "W", "fear", "N"],
    )