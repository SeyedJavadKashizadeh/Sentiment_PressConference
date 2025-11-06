import numpy as np
import soundfile as sf

import librosa
import opensmile
from huggingface_hub import HfFileSystem, list_repo_files, hf_hub_download

REPO_ID = "FedSentimentLab/Fed_audio_text_video"

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

def get_audio_files_by_folder() -> dict:
    """
    Retrieve all audio file paths grouped by folder from a Hugging Face dataset.

    This function uses the Hugging Face Hub's filesystem API to list directories and files
    under the "audio_files_split" folder of the "FedSentimentLab/Fed_audio_text_video" dataset.
    It returns a dictionary where each key is a folder name (relative to "audio_files_split")
    and each value is a list of audio file paths within that folder (relative to the dataset root).

    Returns:
    --------
    dict
        A dictionary mapping folder names (str) to lists of audio file paths (str) contained in each folder.
    """
    
    audio_files_per_folder = {}

    fs = HfFileSystem()
    base_dataset_path = "datasets/FedSentimentLab/Fed_audio_text_video/"
    audio_split_path = base_dataset_path + "audio_files_split/"

    folders = [
        item['name'] 
        for item in fs.ls(audio_split_path, detail=True) 
        if item['type'] == 'directory'
    ]

    for folder_path in folders:
        folder_contents = fs.ls(folder_path, detail=True)
        audio_files = [
            item['name'].replace(base_dataset_path, "") 
            for item in folder_contents 
            if item['type'] == 'file'
        ]
        folder_name = folder_path.replace(audio_split_path, "")
        audio_files_per_folder[folder_name] = audio_files

    return audio_files_per_folder

from huggingface_hub import hf_hub_download

def download_audio_files_hf(hf_token: str, files_by_folder: dict, download_one: bool) -> dict:
    """
    Download audio files from a Hugging Face dataset repository.

    This function downloads files organized by folder from a specified Hugging Face dataset repository.
    Depending on the `download_one` flag, it either downloads all files from the first folder in the dictionary
    or downloads all files from all folders.

    Parameters:
    ----------
    token : str
        Authentication token for private Hugging Face repositories. Use None if the repo is public.
    files_by_folder : dict
        A dictionary mapping folder names to lists of file paths (relative to the dataset root) 
        that need to be downloaded.
    download_one : bool
        If True, download only files from the first folder in the dictionary.
        If False, download files from all folders in the dictionary.

    Returns:
    -------
    dict
        A dictionary mapping each downloaded file's relative path to its local downloaded path.
    """

    local_paths = {}

    if download_one:
        first_key = list(files_by_folder.keys())[0]
        for file in files_by_folder[first_key]:
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=file,
                    token=hf_token
                )
                local_paths[file] = local_path
                print(f"Downloaded {file} to {local_path}")
            except Exception as e:
                print(f"Failed to download {file}: {e}")
    else:
        for key in files_by_folder:
            for file in files_by_folder[key]:
                try:
                    local_path = hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        filename=file,
                        token=hf_token
                    )
                    local_paths[file] = local_path
                    print(f"Downloaded {file} to {local_path}")
                except Exception as e:
                    print(f"Failed to download {file}: {e}")

    return local_paths
