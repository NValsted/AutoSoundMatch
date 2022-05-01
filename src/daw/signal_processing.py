from functools import reduce
from multiprocessing import Pool
from typing import Callable, Optional, Union

import numpy as np
import torch

from src.config.base import PYTORCH_DEVICE, REGISTRY


class SignalProcessor:
    """
    Processes an audio signal, returning features ready to be fed to a
    model.
    """

    _processor: torch.nn.Sequential

    def __init__(self, safe: bool = True, *args, **kwargs):
        if safe:
            self.forward = self._safe_forward
        else:
            self.forward = self._fast_forward

        pipeline = REGISTRY.SIGNAL_PROCESSING.pipeline
        self._processor = torch.nn.Sequential(*pipeline).to(PYTORCH_DEVICE)

    fit: Optional[Callable] = REGISTRY.SIGNAL_PROCESSING.fit

    def _safe_forward(self, signal: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float().to(PYTORCH_DEVICE)

        if signal.min() == signal.max():
            raise ValueError(f"Signal is constant - {signal.shape=}")

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

    @staticmethod
    def concurrent_batch_process(
        signals: list[torch.Tensor], num_workers: int = 4
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
) -> torch.Tensor:
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
) -> torch.Tensor:
    """
    Computes the mean squared error of two signals.
    """
    source = SIGNAL_PROCESSOR(source)
    target = SIGNAL_PROCESSOR(target)

    squared_diff = torch.pow(target - source, 2)
    mse = torch.mean(squared_diff)
    return mse


def silent_signal(
    signal: Union[np.ndarray, torch.Tensor], threshold: float = 1e-04
) -> bool:
    if signal.max() - signal.min() < threshold:
        return True
    return False
