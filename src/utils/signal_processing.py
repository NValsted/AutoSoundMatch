from typing import Union

import librosa
import numpy as np
import torch

from src.config.base import PYTORCH_DEVICE, REGISTRY


def stereo_to_mono(signal: np.ndarray):
    """
    Converts a stereo signal to mono.
    """
    return np.mean(signal, axis=1)


def process_sample(signal: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Processes an audio signal, returning features ready to be fed to a
    model.
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    if signal.shape[-1] == 2:
        signal = stereo_to_mono(signal)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=REGISTRY.SYNTH.sample_rate,
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
    source = torch.from_numpy(process_sample(source)).float().to(PYTORCH_DEVICE)
    target = torch.from_numpy(process_sample(target)).float().to(PYTORCH_DEVICE)

    squared_diff = torch.pow(target - source, 2)
    euclidean_norm = torch.sqrt(torch.sum(squared_diff))
    normalization_factor = 1 / torch.sqrt(torch.sum(torch.pow(target, 2)))
    spectral_convergence = torch.mean(euclidean_norm * normalization_factor)
    return spectral_convergence


def spectral_mse(
    source: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]
):
    """
    Computes the mean squared error of two signals.
    """
    source = torch.from_numpy(process_sample(source)).float().to(PYTORCH_DEVICE)
    target = torch.from_numpy(process_sample(target)).float().to(PYTORCH_DEVICE)

    squared_diff = torch.pow(target - source, 2)
    mse = torch.mean(squared_diff)
    return mse
