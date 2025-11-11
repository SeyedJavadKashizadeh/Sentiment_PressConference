"""
MAIN IDEA
---------
A structured parser and filtering engine for the EmoDB emotional speech
dataset. The module extracts metadata directly from the compact EmoDB filename
stems (e.g., "03a01Fa.wav"), parses speaker codes and emotion codes, and
represents each audio file as a strongly typed pair (`EmoId`, `EmoFile`) for
clean dataset manipulation and machine-learning workflows.

The parser is fully robust to filename variations and performs strict
validation, ensuring only valid EmoDB files enter the pipeline.

MAIN TYPES
----------
- EmoId:
    - Represents a parsed EmoDB filename.
    - Parsed fields:
        * speaker_code  → e.g. "03", "16"
        * emotion_id    → normalized ID (1,3,4,5,6,7,9)
    - Derived properties:
        * gender → "male"/"female"/"unknown"
        * emotion_code → "N", "F", "T", "W", "A", "E", "L"
        * emotion_name → same as emotion_code (matches dataset naming)
    - Factory:
        * EmoId.from_stem(stem) extracts and validates fields using regex

- EmoFile:
    - Couples a filesystem Path with its parsed EmoId.
    - Factory:
        * EmoFile.from_path(path)
    - Filtering via:
        * .matches(gender=..., emotion=..., speaker_code_in={...}, emotion_code_in={...})

FILTERING
---------
The `.matches(**criteria)` method supports:

- Exact matches:
      gender="female"
      speaker_code="16"
      emotion=4
      emotion_code="W"

- Set membership using *_in:
      speaker_code_in={"03","10","11"}
      emotion_in={3,5,7}
      emotion_code_in={"F","W"}

- Unknown keys raise ValueError.

Filtering works across all parsed attributes from the EmoId dataclass.

SCANNING & SELECTION
--------------------
- scan_emodb(root, pattern="*.wav"):
      Recursively collects WAV files under the dataset root, parses each using
      EmoFile.from_path(), and silently skips files not matching the EmoDB
      naming scheme.

- select(files, **criteria):
      Convenience wrapper applying `.matches()` across the collection.

EXAMPLE USAGE
-------------
>>> root = Path("/path/to/EmoDB")
>>> all_files = scan_emodb(root)
>>> angry_female = select(
...     all_files,
...     gender="female",
...     emotion_code="W"   # anger
... )

ENTRY POINT
-----------
main(root, ..., emotion_in=None, emotion_code_in=None) → List[EmoFile]

Performs:
    1. Dataset scan
    2. Criteria dictionary construction
    3. Filtering
    4. Summary printing
    5. Returns the filtered subset

NOTES & GUARANTEES
------------------
- Parsing is strictly regex-based:
      ^(?P<speaker>\d{2})([A-Za-z])\d{2}(?P<emotion>[A-Za-z])[A-Za-z]?$
  guaranteeing that only valid EmoDB stems are accepted.

- Speaker gender is inferred from predefined speaker code lists.

- Emotion codes are mapped to numeric standardized IDs using
  EMOTION_CODE_TO_ID.

- All dataclasses are immutable (`frozen=True`), ensuring safety and
  reproducibility in ML pipelines.

OUTPUT LIST EXAMPLE
-------------------
Example parsed objects:

EmoFile(
    path=.../16b10Td.wav,
    eid=EmoId(speaker_code='16', emotion_id=4)
)
EmoFile(
    path=.../16b10Wa.wav,
    eid=EmoId(speaker_code='16', emotion_id=5)
)
EmoFile(
    path=.../16b10Wb.wav,
    eid=EmoId(speaker_code='16', emotion_id=5)
)

These correspond to speaker 16 (female), expressing:
    - "T" → sadness (emotion 4)
    - "W" → anger  (emotion 5)
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import re

#########
# Enums #
#########

EMOTION_ID_TO_CODE: Dict[int, str] = {
    1: "N",  # neutral
    3: "F",  # happiness (Freude)
    4: "T",  # sadness (Trauer)
    5: "W",  # anger (Wut)
    6: "A",  # fear (Angst)
    7: "E",  # disgust (Ekel)
    9: "L",  # boredom (Langeweile)
}

EMOTION_CODE_TO_ID = {v: k for k, v in EMOTION_ID_TO_CODE.items()}

MALE_SPEAKER_CODES = {"03", "10", "11", "12", "15"}
FEMALE_SPEAKER_CODES = {"08", "09", "13", "14", "16"}
ALL_SPEAKER_CODES = MALE_SPEAKER_CODES | FEMALE_SPEAKER_CODES

##############################
# String normalization utils #
##############################

_STEM_RE = re.compile(r"^(?P<speaker>\d{2})([A-Za-z])\d{2}(?P<emotion>[A-Za-z])[A-Za-z]?$")

def _parse_stem(stem: str) -> tuple[str, str]:
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Invalid EmoDB filename stem: {stem}")
    speaker = m.group("speaker")
    emo_code = m.group("emotion").upper()
    return speaker, emo_code

###############
# Dataclasses #
###############

@dataclass(frozen=True)
class EmoId:
    speaker_code: str   # "03", "08", ...
    emotion_id: int     # normalized integer ID (1,3,4,5,6,7,9)

    @property
    def gender(self) -> str:
        if self.speaker_code in MALE_SPEAKER_CODES:
            return "male"
        if self.speaker_code in FEMALE_SPEAKER_CODES:
            return "female"
        return "unknown"

    @property
    def emotion_code(self) -> str:
        return EMOTION_ID_TO_CODE[self.emotion_id]

    @property
    def emotion_name(self) -> str:
        return EMOTION_ID_TO_CODE[self.emotion_id]

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
          - emotion=<int id> or emotion_in={...}
          - emotion_code="F" or emotion_code_in={"F","W"}
        """
        for key, value in criteria.items():
            if key == "gender":
                if self.eid.gender != value:
                    return False
            elif key == "speaker_code":
                if self.eid.speaker_code != value:
                    return False
            elif key == "emotion":
                if self.eid.emotion_id != int(value):
                    return False
            elif key == "emotion_code":
                if self.eid.emotion_code != str(value).upper():
                    return False
            elif key.endswith("_in"):
                field = key[:-3]
                if field == "gender":
                    attr = self.eid.gender
                elif field == "speaker_code":
                    attr = self.eid.speaker_code
                elif field == "emotion":
                    attr = self.eid.emotion_id
                elif field == "emotion_code":
                    attr = self.eid.emotion_code
                else:
                    raise ValueError(f"Unknown filter key: {key}")
                if attr not in value:
                    return False
            else:
                raise ValueError(f"Unknown filter key: {key}")
        return True

#############
# Utilities #
#############
def scan_emodb(root: Path, pattern: str = "*.wav") -> List[EmoFile]:
    files: List[EmoFile] = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            files.append(EmoFile.from_path(p))
        except Exception:
            # Skip files that don't match the scheme
            print(p)
            continue
    return files

def select(files: Iterable[EmoFile], **criteria: Any) -> List[EmoFile]:
    return [f for f in files if f.matches(**criteria)]

def main(
    root: Path,
    gender: Optional[str] = None,
    speaker_code: Optional[str] = None,
    speaker_code_in: Optional[Iterable[str]] = None,
    emotion: Optional[int] = None,
    emotion_in: Optional[Iterable[int]] = None,
    emotion_code: Optional[str] = None,
    emotion_code_in: Optional[Iterable[str]] = None,
) -> List[EmoFile]:
    """Scan EmoDB under `root` and return files matching the given criteria."""
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
        criteria["emotion"] = int(emotion)
    if emotion_in is not None:
        criteria["emotion_in"] = {int(x) for x in emotion_in}

    if emotion_code is not None:
        criteria["emotion_code"] = emotion_code.upper()
    if emotion_code_in is not None:
        criteria["emotion_code_in"] = {x.upper() for x in emotion_code_in}

    selected = select(all_files, **criteria)
    print(f"Selected files: {len(selected)}")
    for f in selected[:5]:
        print(" -", f.path.name)
    return selected

if __name__ == "__main__":
    # Example local test
    root = Path(__file__).resolve().parents[2]
    root = root / "dataset_training" / "EmoDB"
    female_happy = main(root)
    breakpoint()