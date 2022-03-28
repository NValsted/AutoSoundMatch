from functools import partial, reduce
from multiprocessing import Pool
from typing import Tuple, Union

import numpy as np
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from src.config.base import PYTORCH_DEVICE, REGISTRY


class StereoToMono(torch.nn.Module):
    """
    Conditionally converts a stereo signal to mono.
    """

    _processor: torch.nn.Module

    def __init__(self, signal_shape: Tuple):
        super(StereoToMono, self).__init__()
        if signal_shape[-1] == 2:
            self._processor = partial(torch.mean, axis=-1)
        else:
            self._processor = torch.nn.Identity()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self._processor(signal)


class MinMax(torch.nn.Module):
    """
    Min-max mormalizes a signal to a given amplitude range.
    """

    _processor: torch.nn.Module
    amplitude_min: int
    amplitude_max: int

    def __init__(self, amplitude_min: int = -1, amplitude_max: int = 1):
        super(MinMax, self).__init__()
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return (signal - signal.min()) / (signal.max() - signal.min()) * (
            self.amplitude_max - self.amplitude_min
        ) + self.amplitude_min


class SignalProcessor:
    """
    Processes an audio signal, returning features ready to be fed to a
    model.
    """

    _processor: torch.nn.Sequential

    def __init__(
        self, safe: bool = True, signal_shape: Tuple = (None, 2), *args, **kwargs
    ):
        if safe:
            self.forward = self._safe_forward
        else:
            self.forward = self._fast_forward

        pipeline = (
            StereoToMono(signal_shape),
            MinMax(),
            MelSpectrogram(  # TODO: parameterize through REGISTRY
                sample_rate=REGISTRY.SYNTH.sample_rate,
                n_fft=2048,
                n_mels=128,
                hop_length=1024,
                f_min=30,
                f_max=11000,
                pad=0,
            ),
            AmplitudeToDB(top_db=80),
        )
        self._processor = torch.nn.Sequential(*pipeline).to(PYTORCH_DEVICE)

    def _safe_forward(self, signal: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float().to(PYTORCH_DEVICE)

        if signal.min() == 0 and signal.max() == 0:
            raise ValueError("Signal is all zeros")

        return self._processor(signal)

    def _fast_forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self._processor(signal)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @classmethod
    def batch_process(cls, signals: list[torch.Tensor]) -> list[torch.Tensor]:
        processor = cls()
        processed_signals = [processor(signal) for signal in signals]
        torch.cuda.empty_cache()
        return processed_signals

    @classmethod
    def concurrent_batch_process(
        cls, signals: list[torch.Tensor], num_workers: int = 4
    ) -> list[torch.Tensor]:

        chunks = [
            signals[i : i + num_workers]  # NOQA: E203
            for i in range(0, len(signals), num_workers)
        ]

        with Pool(num_workers) as pool:
            processed_chunks = pool.map(SignalProcessor.batch_process, chunks)

        collected = reduce(lambda x, y: x + y, processed_chunks, [])
        return collected


SIGNAL_PROCESSOR = SignalProcessor()


def spectral_convergence(
    source: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]
):
    """
    Computes the spectral convergence of two signals, i.e. the mean magnitude-normalized
    Euclidean norm - Esling, Philippe, et al. (2019).
    """
    source = SIGNAL_PROCESSOR(source)
    target = SIGNAL_PROCESSOR(target)

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
    source = SIGNAL_PROCESSOR(source)
    target = SIGNAL_PROCESSOR(target)

    squared_diff = torch.pow(target - source, 2)
    mse = torch.mean(squared_diff)
    return mse
