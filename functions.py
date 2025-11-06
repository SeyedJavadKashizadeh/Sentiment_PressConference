import numpy as np
import soundfile as sf

import librosa
import opensmile

def extract_features_librosa(file_name: str) -> np.ndarray:
    """
    Extract audio features explicitly specifying frequency parameters:
        - 40 MFCCs (based on 128 Mel bands)
        - 12 Chroma coefficients
        - 128 Mel-spectrogram frequencies
        - Spectral Contrast
        - Tonnetz
    All features are computed with fmin=0 Hz, fmax=8000 Hz (Nyquist for 16kHz audio)
    """

    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # Short-time Fourier transform
        stft = np.abs(librosa.stft(X))

        # ---- MFCCs (40 coefficients from 128 Mel bands) ----
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=X,
                sr=sample_rate,
                n_mfcc=40,
                n_mels=128,
                fmin=0,
                fmax=8000
            ).T,
            axis=0
        )

        # ---- Chroma (12 coefficients) ----
        chroma = np.mean(
            librosa.feature.chroma_stft(
                y=X,
                sr=sample_rate,
                n_chroma=12,
                n_fft=4096
            ).T,
            axis=0
        )

        # ---- Mel-spectrogram (128 bands, 0–8kHz) ----
        mel = np.mean(
            librosa.feature.melspectrogram(
                y=X,
                sr=sample_rate,
                n_mels=128,
                fmin=0,
                fmax=8000
            ).T,
            axis=0
        )

        # ---- Spectral Contrast ----
        contrast = np.mean(
            librosa.feature.spectral_contrast(
                S=stft,
                sr=sample_rate,
                fmin=200.0
            ).T,
            axis=0
        )

        # ---- Tonnetz ----
        tonnetz = np.mean(
            librosa.feature.tonnetz(
                y=librosa.effects.harmonic(X),
                sr=sample_rate
            ).T,
            axis=0
        )

        features = np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
        features = features.reshape(1, -1)

    return features

def extract_features_opensmile(file_name: str) -> np.ndarray:
    """
    Extract audio features via OpenSmile
    """

    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    #feature_level=opensmile.FeatureLevel.LowLevelDescriptors # gives time series of features
)

    features = smile.process_file(file_name)
    features = features.to_numpy()

    return features