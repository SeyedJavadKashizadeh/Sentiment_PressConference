"""
MAIN IDEA
---------
A lightweight, type-safe toolkit for parsing, normalizing, scanning, and
filtering the TESS (Toronto Emotional Speech Set) dataset. It extracts speaker
and emotion information from directory names such as "OAF_happy" and represents
each audio file as a structured object (`TessId`, `TessFile`) to support
clean, flexible dataset manipulation.

MAIN TYPES
----------
- TessId:
    - Holds parsed metadata (speaker + emotion ID)
    - Speaker: "OAF" or "YAF"
    - Emotion: integer ID 1..8
    - Properties: age_group, emotion_name
    - Factory: from_dirname(dirname) → splits at the first underscore only

- TessFile:
    - Binds a file path to a TessId
    - Factory: from_path(path) parses the parent folder name

FILTERING
---------
- Implemented through TessFile.matches(**criteria)
- Supports:
    - Exact matches: speaker="OAF", emotion=3
    - Emotion by name or ID (normalized automatically)
    - Set membership using *_in keys: speaker_in={"OAF","YAF"},
      emotion_in={1, 3, "pleasant_surprise"}
- Unknown keys raise ValueError

SCANNING & SELECTION
--------------------
- scan_tess(root, pattern="*.wav"):
    Recursively gathers WAV files under `root`, parses each into a TessFile,
    ignores malformed folders.

- select(files, **criteria):
    Wrapper around .matches() for clean high-level filtering.

EXAMPLE USAGE
-------------
>>> root = Path(__file__).resolve().parents[2]
>>> all_files = scan_tess(root)
>>> happy_like = select(
...     all_files,
...     speaker="YAF",
...     emotion_in={"happy", "pleasant-surprise"},
... )

NOTES & GUARANTEES
------------------
- Directory parsing uses dirname.split("_", 1) to avoid breaking emotion names
  containing underscores or hyphens.
- Normalization maps variants like "Fear", "fearful", "pleasant-surprise" to
  canonical emotion names.
- Dataclasses are frozen for immutability and safe hashing.
- scan_tess() is robust: invalid folder names are skipped gracefully.

OUTPUT LIST EXAMPLE
-------------------
TessFile(path=.../OAF_happy/OAF_yes_happy.wav,
         tid=TessId(speaker='OAF', emotion=3))
TessFile(path=.../OAF_happy/OAF_young_happy.wav,
         tid=TessId(speaker='OAF', emotion=3))
TessFile(path=.../OAF_happy/OAF_youth_happy.wav,
         tid=TessId(speaker='OAF', emotion=3))

All files share: speaker=OAF, emotion=3 ("happy").
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Union
import re

#########
# Enums #
#########

EMOTION: Dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "pleasant_surprise",
}

NAME_NORMALIZE = {
    "neutral": "neutral",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "fearful": "fear",
    "disgust": "disgust",
    "pleasant_surprise": "pleasant_surprise",
    "pleasant-surprise": "pleasant_surprise",
    "pleasantsurprise": "pleasant_surprise",
}

EMOTION_NAME_TO_ID = {v: k for k, v in EMOTION.items()}

##############################
# String normalization utils #
##############################

def _norm_emotion_name(s: str) -> str:
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z_]", "", s)
    return NAME_NORMALIZE.get(s, s)

###############
# Dataclasses #
###############

@dataclass(frozen=True)
class TessId:
    speaker: str    # "OAF" or "YAF"
    emotion: int    # 1..8

    @property
    def age_group(self) -> str:
        return "younger_adult_female" if self.speaker == "YAF" else "older_adult_female"

    @property
    def emotion_name(self) -> str:
        return EMOTION[self.emotion]

    @classmethod
    def from_dirname(cls, dirname: str) -> "TessId":
        parts = dirname.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid TESS directory name: {dirname}")

        speaker, emo_raw = parts
        if speaker not in {"OAF", "YAF"}:
            raise ValueError(f"Unknown speaker code in {dirname}")

        emo_norm = _norm_emotion_name(emo_raw)
        if emo_norm not in EMOTION_NAME_TO_ID:
            raise ValueError(f"Unknown emotion '{emo_raw}' in {dirname}")

        return cls(speaker=speaker, emotion=EMOTION_NAME_TO_ID[emo_norm])


@dataclass(frozen=True)
class TessFile:
    path: Path
    tid: TessId

    @classmethod
    def from_path(cls, path: Path) -> "TessFile":
        tid = TessId.from_dirname(path.parent.name)
        return cls(path=path, tid=tid)

    def matches(self, **criteria: Any) -> bool:
        for key, val in criteria.items():

            # emotion filter (accept id or name)
            if key == "emotion":
                if isinstance(val, str):
                    val = EMOTION_NAME_TO_ID[_norm_emotion_name(val)]
                if self.tid.emotion != val:
                    return False

            elif key.endswith("_in"):
                field = key[:-3]
                if field == "emotion":
                    allowed_ids = {
                        EMOTION_NAME_TO_ID[_norm_emotion_name(v)] if isinstance(v, str) else v
                        for v in val
                    }
                    if self.tid.emotion not in allowed_ids:
                        return False
                else:
                    attr = getattr(self.tid, field)
                    if attr not in val:
                        return False

            else:
                if not hasattr(self.tid, key):
                    raise ValueError(f"Unknown filter key: {key}")
                if getattr(self.tid, key) != val:
                    return False

        return True

#############
# Utilities #
#############

def scan_tess(root: Path, pattern: str = "*.wav") -> List[TessFile]:
    files: List[TessFile] = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            files.append(TessFile.from_path(p))
        except Exception:
            continue
    return files


def select(files: Iterable[TessFile], **criteria: Any) -> List[TessFile]:
    return [f for f in files if f.matches(**criteria)]

############
#   Main   #
############

def main(
    root: Path,
    speaker: Optional[str] = None,
    emotion: Optional[Union[int, str]] = None,
    speaker_in: Optional[Iterable[str]] = None,
    emotion_in: Optional[Iterable[Union[int, str]]] = None,
) -> List[TessFile]:

    # Scan dataset
    all_files = scan_tess(root)
    print(f"Total parsed files: {len(all_files)}")

    criteria: Dict[str, Any] = {}

    # Filters
    if speaker is not None:
        criteria["speaker"] = speaker.upper()

    if speaker_in is not None:
        criteria["speaker_in"] = {s.upper() for s in speaker_in}

    if emotion is not None:
        criteria["emotion"] = emotion

    if emotion_in is not None:
        criteria["emotion_in"] = emotion_in

    # Apply filters
    selected = select(all_files, **criteria)

    print(f"Selected files: {len(selected)}")
    for f in selected[:10]:
        print(" -", f.path.relative_to(root))

    return selected

#############
# For testing only
#############

if __name__ == "__main__":
    # Example local test
    root = Path(__file__).resolve().parents[2]
    root = root / "dataset_training" / "TESS"
    selected = main(root, speaker="OAF", emotion="happy")