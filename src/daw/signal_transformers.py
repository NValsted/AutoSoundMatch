from functools import partial

import torch


class StereoToMono(torch.nn.Module):
    """
    Conditionally converts a stereo signal to mono.
    """

    _processor: torch.nn.Module

    def __init__(self, signal_shape: tuple):
        super(StereoToMono, self).__init__()
        if signal_shape[-1] == 2:
            self._processor = partial(torch.mean, axis=-1)
        else:
            self._processor = torch.nn.Identity()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self._processor(signal)


class MinMax(torch.nn.Module):
    """
    Min-max normalizes a signal to a given amplitude range.
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


class LogTransform(torch.nn.Module):
    # References https://github.com/acids-ircam/flow_synthesizer/blob/47abcca360ea3e2a4104855e30d6e548d207e802/code/utils/transforms.py  # NOQA : E501
    clip: float

    def __init__(self, clip: float = 1e-3):
        super(LogTransform, self).__init__()
        self.clip = clip

    def forward(self, data: torch.Tensor):
        if self.clip == 0.0:
            data = torch.log1p(data)
        else:
            data = torch.log(data + self.clip)
        return data


class ZScoreNormalize(torch.nn.Module):
    # References https://github.com/acids-ircam/flow_synthesizer/blob/47abcca360ea3e2a4104855e30d6e548d207e802/code/utils/transforms.py  # NOQA : E501

    def __init__(self, mean, std):
        super(ZScoreNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, data: torch.Tensor):
        if type(self.mean) == float:
            data = (data - self.mean) / self.std
        else:
            for c in range(data.shape[0]):
                data[c] = (data[c] - self.mean[c]) / self.std[c]
        return data
