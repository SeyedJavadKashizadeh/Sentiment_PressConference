"""
utils.py
========

Unified emotion normalization utilities for all speech-emotion datasets
(EmoDB, RAVDESS, TESS) used in the training pipeline.

This module defines the **universal emotion ID space**, provides helpers for
converting raw dataset emotion labels into this shared scheme, and exposes
dataset-specific mapping functions.

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------

Many emotion datasets use different coding systems:

• **RAVDESS**  
    Uses integer codes 1–8 (neutral, calm, happy, sad, angry, fearful,
    disgust, pleasant surprise).

• **TESS**  
    Uses folder names (“YAF_happy”, “OAF_sad”), often inconsistent casing,
    hyphens, etc.

• **EmoDB**  
    Uses **single-letter codes** (N, F, T, W, A, E, L), some of which do not
    correspond directly to RAVDESS or TESS labels.

To train unified models across datasets, we map every label into one
**canonical emotion ID space**:

    ID → Name  
    1  → neutral  
    2  → calm  
    3  → happy  
    4  → sad  
    5  → angry  
    6  → fear  
    7  → disgust  
    8  → pleasant_surprise  
    9  → boredom  

----------------------------------------------------------------------
CALM COLLAPSED INTO NEUTRAL
----------------------------------------------------------------------

For cross-dataset consistency, *this module collapses “calm” (ID=2)
into “neutral” (ID=1)*.

This ensures that:

• EmoDB: has no "calm" → no mismatch  
• TESS: does not provide a calm category  
• RAVDESS calm samples merge into the neutral class  
• Downstream models see a simplified and dataset-aligned class structure  

This collapse is implemented **inside the universal normalizer**:

    normalize_universal_emotion("calm")  → 1  
    normalize_universal_emotion(2)       → 1  

----------------------------------------------------------------------
MAIN FUNCTIONS
----------------------------------------------------------------------

normalize_universal_emotion(x)
    Convert a raw emotion (ID, name, alias) into its canonical ID.
    Handles messy strings, casing, hyphens, underscores.
    Applies CALM → NEUTRAL collapse.

normalize_universal_emotion_set(vals)
    Vectorized convenience wrapper for lists/sets.

normalize_emodb_emotion(x)
    Normalize EmoDB emotion codes ("N", "F", "W", etc.) or names to
    universal IDs.

normalize_ravdess_emotion(x)
normalize_tess_emotion(x)
    Normalize label values for RAVDESS or TESS into the universal ID space.

----------------------------------------------------------------------
DATASET-SPECIFIC METADATA
----------------------------------------------------------------------

The module also exposes:

• RAVDESS_STATEMENT  
• RAVDESS_MODALITY  
• RAVDESS_VOCAL_CHANNEL  
• RAVDESS_EMOTION_INTENSITY  

and mapping dictionaries for **EmoDB**:

• EMODB_CODE_TO_UNIV  
• EMODB_UNIV_TO_CODE  

These are used by dataset parsers to ensure consistent labeling and metadata
interpretation.

----------------------------------------------------------------------
WHY THIS MODULE EXISTS
----------------------------------------------------------------------

Without a unified mapping, training a classifier across datasets produces:

- label mismatches  
- missing classes  
- duplicated semantics (neutral vs calm)  
- inconsistent file filtering in the CLI  

By centralizing all normalization logic here, every downstream parser:

- becomes shorter, cleaner, and dataset-focused  
- speaks a shared emotion language  
- automatically benefits from future adjustments to label mapping  

This is the authoritative ground truth for all emotion-space semantics used
in the project.

----------------------------------------------------------------------
"""
from __future__ import annotations
from typing import Union, Iterable, Dict
import re

###############################################################################
# Canonical Universal Emotion Space (Internal Representation)
###############################################################################

UNIVERSAL_EMOTION_ID_TO_NAME: Dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "pleasant_surprise",
    9: "boredom"
}

UNIVERSAL_NAME_TO_ID = {v: k for k, v in UNIVERSAL_EMOTION_ID_TO_NAME.items()}

# Strong aliases shared across datasets
UNIVERSAL_ALIASES = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "fear": "fear",
    "fearful": "fear",
    "disgust": "disgust",
    "surprise": "pleasant_surprise",
    "pleasant_surprise": "pleasant_surprise",
    "pleasant-surprise": "pleasant_surprise",
    "pleasantsurprise": "pleasant_surprise",
    "boredom": "boredom",
}


def _normalize_string(s: str) -> str:
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z_]", "", s)
    return s


###############################################################################
# Universal Emotion Normalization
###############################################################################

def normalize_universal_emotion(x: Union[int, str]) -> int:
    """
    Convert an emotion into the canonical universal ID,
    with 'calm' collapsed to 'neutral' (2 -> 1).
    """
    if isinstance(x, int):
        if x not in UNIVERSAL_EMOTION_ID_TO_NAME:
            raise ValueError(f"Unknown emotion id {x}")
        # hard collapse numeric calm (2) to neutral (1)
        return 1 if x == 2 else x

    if isinstance(x, str):
        s = _normalize_string(x)
        if s in UNIVERSAL_ALIASES:
            s = UNIVERSAL_ALIASES[s]  # 'calm' -> 'neutral'
        if s not in UNIVERSAL_NAME_TO_ID:
            raise ValueError(f"Unknown emotion name: {x}")
        eid = UNIVERSAL_NAME_TO_ID[s]
        return 1 if eid == 2 else eid

    raise TypeError(f"Emotion must be int or str (got {type(x)})")


def normalize_universal_emotion_set(vals: Iterable[Union[int, str]]) -> set[int]:
    return {normalize_universal_emotion(v) for v in vals}


###############################################################################
# Dataset-Specific Mappings
###############################################################################
# RAVDESS
RAVDESS_STATEMENT = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}
RAVDESS_MODALITY = {1: "full-AV", 2: "video-only", 3: "audio-only"}
RAVDESS_VOCAL_CHANNEL = {1: "speech", 2: "song"}
RAVDESS_EMOTION_INTENSITY = {1: "normal", 2: "strong"}

# EmoDB codes
EMODB_CODE_TO_UNIV = {"N": 1, "F": 3, "T": 4, "W": 5, "A": 6, "E": 7, "L": 9}
# (EmoDB uses ID=9 for boredom → you may decide to map this to universal 8 or keep distinct)
EMODB_UNIV_TO_CODE = {v: k for k, v in EMODB_CODE_TO_UNIV.items()}

###############################################################################
# Dataset-Specific Emotion Normalizers
###############################################################################

def normalize_emodb_emotion(x: Union[int, str]) -> int:
    """Normalize from EmoDB's codes or names → universal ID."""
    if isinstance(x, int):
        return normalize_universal_emotion(x)

    x = _normalize_string(x)
    if x.upper() in EMODB_CODE_TO_UNIV:
        return EMODB_CODE_TO_UNIV[x.upper()]
    return normalize_universal_emotion(x)


def normalize_emodb_emotion_set(vals: Iterable[Union[int, str]]) -> set[int]:
    return {normalize_emodb_emotion(v) for v in vals}


def normalize_ravdess_emotion(x: Union[int, str]) -> int:
    return normalize_universal_emotion(x)


def normalize_ravdess_emotion_set(vals: Iterable[Union[int, str]]) -> set[int]:
    return {normalize_ravdess_emotion(v) for v in vals}


def normalize_tess_emotion(x: Union[int, str]) -> int:
    return normalize_universal_emotion(x)


def normalize_tess_emotion_set(vals: Iterable[Union[int, str]]) -> set[int]:
    return {normalize_tess_emotion(v) for v in vals}
