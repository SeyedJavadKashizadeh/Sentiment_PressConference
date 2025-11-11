from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Callable, Optional
import os
#########
# Enums #
#########

EMOTION = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

STATEMENT = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

MODALITY = {1: "full-AV", 2: "video-only", 3: "audio-only"}

VOCAL_CHANNEL = {1: "speech", 2: "song"}

EMOTION_INTENSITY = {1: "normal", 2: "strong"}


###############
# Dataclasses #
###############

@dataclass(frozen=True)
class RavdessId:
    modality: int         # 01..03
    vocal_channel: int    # 01..02
    emotion: int          # 01..08
    intensity: int        # 01..02 (no '02' for emotion==1)
    statement: int        # 01..02
    repetition: int       # 01..02
    actor: int            # 01..24

    @property
    def gender(self) -> str:
        # Odd actor IDs are male, even are female
        return "male" if self.actor % 2 == 1 else "female"

    @property
    def emotion_name(self) -> str:
        return EMOTION.get(self.emotion, f"unknown({self.emotion})")

    @property
    def modality_name(self) -> str:
        return MODALITY.get(self.modality, f"unknown({self.modality})")

    @property
    def vocal_channel_name(self) -> str:
        return VOCAL_CHANNEL.get(self.vocal_channel, f"unknown({self.vocal_channel})")

    @property
    def intensity_name(self) -> str:
        # Neutral (1) has no strong intensity in the dataset
        return EMOTION_INTENSITY.get(self.intensity, f"unknown({self.intensity})")

    @property
    def statement_text(self) -> str:
        return STATEMENT.get(self.statement, f"unknown({self.statement})")

    @classmethod
    def from_stem(cls, stem: str) -> "RavdessId":
        """
        Parse '03-01-06-01-02-01-12' into fields.
        """
        parts = stem.split("-")
        if len(parts) != 7 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Invalid RAVDESS stem: {stem}")
        m, vc, emo, inten, stmt, rep, act = (int(p) for p in parts)
        # Basic validations
        if not (1 <= m <= 3): raise ValueError(f"modality out of range in {stem}")
        if not (1 <= vc <= 2): raise ValueError(f"vocal_channel out of range in {stem}")
        if not (1 <= emo <= 8): raise ValueError(f"emotion out of range in {stem}")
        if not (1 <= inten <= 2): raise ValueError(f"intensity out of range in {stem}")
        if emo == 1 and inten != 1:  # neutral has no strong
            raise ValueError(f"neutral emotion cannot have strong intensity in {stem}")
        if not (1 <= stmt <= 2): raise ValueError(f"statement out of range in {stem}")
        if not (1 <= rep <= 2): raise ValueError(f"repetition out of range in {stem}")
        if not (1 <= act <= 24): raise ValueError(f"actor out of range in {stem}")
        return cls(m, vc, emo, inten, stmt, rep, act)

    def to_stem(self) -> str:
        return f"{self.modality:02d}-{self.vocal_channel:02d}-{self.emotion:02d}-{self.intensity:02d}-{self.statement:02d}-{self.repetition:02d}-{self.actor:02d}"

@dataclass(frozen=True)
class RavdessFile:
    """
    Container that ties a parsed RAVDESS id to an actual file path.
    """
    path: Path
    rid: RavdessId

    @classmethod
    def from_path(cls, path: Path) -> "RavdessFile":
        rid = RavdessId.from_stem(path.stem)
        return cls(path=path, rid=rid)

    def matches(self, **criteria: Any) -> bool:
        """
        Flexible filtering:
        - gender="male" / "female"
        - emotion=6 or emotion_in={3,6}
        - vocal_channel=1 (speech) / 2 (song)
        - modality=3 (audio-only)
        - intensity=1 or intensity_in={1,2}
        - actor_in={1,3,5}
        etc.
        """
        for key, value in criteria.items():
            if key == "gender":
                if self.rid.gender != value:
                    return False
            elif key.endswith("_in"):
                field = key[:-3]
                attr = getattr(self.rid, field)
                if attr not in value:
                    return False
            else:
                # exact field match against RavdessId attributes
                if not hasattr(self.rid, key):
                    raise ValueError(f"Unknown filter key: {key}")
                if getattr(self.rid, key) != value:
                    return False
        return True


#############
# Utilities #
#############

def scan_ravdess(root: Path, pattern: str = "*.wav") -> List[RavdessFile]:
    files: List[RavdessFile] = []
    skip = "audio_speech_actors_01-24"

    for folder, subdirs, filenames in os.walk(root):
        folder_path = Path(folder)

        if folder_path.name == skip:
            subdirs[:] = []
            continue

        for filename in filenames:
            if filename.endswith(".wav"):
                p = folder_path / filename
                try:
                    files.append(RavdessFile.from_path(p))
                except Exception:
                    continue

    return files

def select(files: Iterable[RavdessFile], **criteria: Any) -> List[RavdessFile]:
    """
    Filter a collection of RavdessFile by attribute criteria (see RavdessFile.matches).
    """
    return [f for f in files if f.matches(**criteria)]

def main(
    root: Path,
    gender: Optional[str] = None,
    modality: Optional[int] = None,
    modality_in: Optional[Iterable[int]] = None,
    vocal_channel: Optional[int] = None,
    vocal_channel_in: Optional[Iterable[int]] = None,
    emotion: Optional[int] = None,
    emotion_in: Optional[Iterable[int]] = None,
    intensity: Optional[int] = None,
    intensity_in: Optional[Iterable[int]] = None,
    statement: Optional[int] = None,
    statement_in: Optional[Iterable[int]] = None,
    repetition: Optional[int] = None,
    repetition_in: Optional[Iterable[int]] = None,
    actor: Optional[int] = None,
    actor_in: Optional[Iterable[int]] = None,
) -> List[RavdessFile]:

    # Scan dataset
    all_files = scan_ravdess(root)
    print(f"Total parsed files: {len(all_files)}")

    # Build criteria dictionary dynamically
    criteria: dict[str, Any] = {}

    if gender is not None:
        criteria["gender"] = gender

    if modality is not None:
        criteria["modality"] = modality
    if modality_in is not None:
        criteria["modality_in"] = set(modality_in)

    if vocal_channel is not None:
        criteria["vocal_channel"] = vocal_channel
    if vocal_channel_in is not None:
        criteria["vocal_channel_in"] = set(vocal_channel_in)

    if emotion is not None:
        criteria["emotion"] = emotion
    if emotion_in is not None:
        criteria["emotion_in"] = set(emotion_in)

    if intensity is not None:
        criteria["intensity"] = intensity
    if intensity_in is not None:
        criteria["intensity_in"] = set(intensity_in)

    if statement is not None:
        criteria["statement"] = statement
    if statement_in is not None:
        criteria["statement_in"] = set(statement_in)

    if repetition is not None:
        criteria["repetition"] = repetition
    if repetition_in is not None:
        criteria["repetition_in"] = set(repetition_in)

    if actor is not None:
        criteria["actor"] = actor
    if actor_in is not None:
        criteria["actor_in"] = set(actor_in)

    # Apply selection
    selected_files = select(all_files, **criteria)

    print(f"Selected files: {len(selected_files)}")
    for f in selected_files[:5]:
        print(" -", f.path.name)

    return selected_files