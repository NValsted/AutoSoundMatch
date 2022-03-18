from typing import Union

import librosa
import numpy as np
import torch


def process_sample(signal: np.ndarray, sample_rate: int):
    """
    Processes an audio signal, returning features ready to be fed to a
    model.
    """

    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=2048,  # TODO: parameterize through REGISTRY
        n_mels=128,
        hop_length=1024,
        fmin=30,
        fmax=11000,
    )
    db_transformed = librosa.power_to_db(mel_spectrogram)

    return db_transformed


def spectral_convergence(
    source: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]
):
    """
    Computes the spectral convergence of two signals, i.e. the mean magnitude-normalized
    Euclidean norm - Esling, Philippe, et al. (2019).
    """
    if isinstance(source, np.ndarray):
        source = torch.from_numpy(source).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    squared_diff = torch.pow(target - source, 2)
    euclidean_norm = torch.sqrt(torch.sum(squared_diff))
    normalization_factor = 1 / torch.sqrt(torch.sum(torch.pow(target, 2)))
    spectral_convergence = torch.mean(euclidean_norm * normalization_factor)
    return spectral_convergence


def mse(
    source: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]
):
    """
    Computes the mean squared error of two signals.
    """
    if isinstance(source, np.ndarray):
        source = torch.from_numpy(source).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    squared_diff = torch.pow(target - source, 2)
    mse = torch.mean(squared_diff)
    return mse
