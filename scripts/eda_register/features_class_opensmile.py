"""
Docstring for scripts.eda.features_class_opensmile
================================
Pitch & loudness = arousal and valence markers

Voice quality = tension/roughness vs clarity

MFCCs = timbre differences across emotion

Spectral balance = brightness associated with anger/happiness

Spectral dynamics = articulatory energy variability

Temporal voicing = rhythm and pausing patterns
"""
PITCH_AND_LOUDNESS = [
    "smile_F0semitoneFrom27.5Hz_sma3nz_amean",        # overall pitch height
    "smile_F0semitoneFrom27.5Hz_sma3nz_percentile80.0", # pitch peaks / excitement
    "smile_loudness_sma3_amean",                      # overall intensity
    "smile_loudness_sma3_percentile80.0",             # loud segments / arousal
]

VOICE_QUALITY = [
    "smile_jitterLocal_sma3nz_amean",                 # pitch stability (roughness)
    "smile_shimmerLocaldB_sma3nz_amean",              # amplitude stability
    "smile_HNRdBACF_sma3nz_amean",                    # harmonic-to-noise ratio
]

# MFCCs describe the timbre of speech
MFCC = [
    "smile_mfcc1_sma3_amean",
    "smile_mfcc2_sma3_amean",
    "smile_mfcc3_sma3_amean",
    "smile_mfcc4_sma3_amean",
]

SPECTRAL_BALANCE = [
    "smile_alphaRatioV_sma3nz_amean",                # low–high frequency energy balance
    "smile_hammarbergIndexV_sma3nz_amean",           # spectral brightness
]

SPECTRAL_DYNAMICS = [
    "smile_spectralFlux_sma3_amean",                 # how rapidly the spectrum changes
    "smile_spectralFluxV_sma3nz_amean",
]

TEMPORAL_VOICING = [
    "smile_VoicedSegmentsPerSec",                    # speech continuity / articulation rate
    "smile_MeanVoicedSegmentLengthSec",              # length of voiced chunks
]

OPENSMILE_KEY_FEATURES = [
    "smile_F0semitoneFrom27.5Hz_sma3nz_amean",
    "smile_loudness_sma3_amean",
    "smile_jitterLocal_sma3nz_amean",
    "smile_shimmerLocaldB_sma3nz_amean",
    "smile_HNRdBACF_sma3nz_amean",
    "smile_spectralFlux_sma3_amean",
]

OPENSMILE_KEY_FEATURES_RENAMED = {
    "smile_F0semitoneFrom27.5Hz_sma3nz_amean": "Mean F0 (pitch)",
    "smile_loudness_sma3_amean": "Mean Loudness",
    "smile_jitterLocal_sma3nz_amean": "Jitter (local)",
    "smile_shimmerLocaldB_sma3nz_amean": "Shimmer (dB)",
    "smile_HNRdBACF_sma3nz_amean": "HNR (harmonics-to-noise)",
    "smile_spectralFlux_sma3_amean": "Spectral Flux",
}