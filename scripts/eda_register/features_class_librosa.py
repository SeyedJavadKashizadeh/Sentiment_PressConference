"""
Docstring for scripts.eda.features_class_librosa
================================
MFCCs show how emotional states modify vocal timbre.

Chroma reflects differences in melodic contour / intonation.

Mel bands reveal how energy shifts across low/mid/high frequencies depending on arousal and valence.

Spectral contrast distinguishes expressive vs flat emotions.

Tonnetz shows harmonic movement differences related to emotional prosody.
"""
MFCC_SUMMARY = [
    "mfcc_mean",
    "mfcc_std",
]

MEL_SUMMARY = [
    "mel_low_mean",
    "mel_mid_mean",
    "mel_high_mean",
]

CHROMA_SUMMARY = [
    "chroma_mean",
    "chroma_std",
]

CONTRAST_SUMMARY = [
    "contrast_mean",
]

TONNETZ_SUMMARY = [
    "tonnetz_mean",
]