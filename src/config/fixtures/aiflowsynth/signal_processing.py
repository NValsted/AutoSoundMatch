from typing import TYPE_CHECKING

from torchaudio.transforms import MelSpectrogram

from src.config.registry_sections import SignalProcessingSection
from src.daw.signal_transformers import LogTransform, StereoToMono, ZScoreNormalize

if TYPE_CHECKING:
    from src.daw.audio_model import AudioBridgeTable
    from src.daw.signal_processing import SignalProcessor


def fit(self: "SignalProcessor", audio_bridges: list["AudioBridgeTable"]):
    import torch
    from torchaudio.transforms import MelSpectrogram

    from src.config.base import PYTORCH_DEVICE
    from src.daw.signal_transformers import LogTransform, StereoToMono, ZScoreNormalize

    tmp_pipeline = (
        StereoToMono((None, 2)),
        MelSpectrogram(
            sample_rate=22050,
            n_fft=2048,
            n_mels=128,
            hop_length=1024,
            f_min=30,
            f_max=11000,
            pad=0,
        ),
        LogTransform(),
    )
    tmp_processor = torch.nn.Sequential(*tmp_pipeline).to(PYTORCH_DEVICE)

    # References https://github.com/acids-ircam/flow_synthesizer/blob/47abcca360ea3e2a4104855e30d6e548d207e802/code/utils/data.py  # NOQA : E501
    mean = 0
    std = 0
    for i, bridge in enumerate(audio_bridges):
        signal = torch.load(bridge.audio_path, map_location=PYTORCH_DEVICE)
        processed = tmp_processor(signal)
        tmp_mean = processed.mean()
        tmp_dev = processed - mean

        mean += (tmp_mean - mean) / (i + 1)
        std += ((processed - mean) * tmp_dev).mean()

    mean = float(mean)
    std = float((std / len(audio_bridges)) ** (1 / 2))

    pipeline = (*tmp_pipeline, ZScoreNormalize(mean, std))
    self._processor = torch.nn.Sequential(*pipeline).to(PYTORCH_DEVICE)


signal_processing_section = SignalProcessingSection(
    pipeline=(
        StereoToMono((None, 2)),
        MelSpectrogram(
            sample_rate=22050,
            n_fft=2048,
            n_mels=128,
            hop_length=1024,
            f_min=30,
            f_max=11000,
            pad=0,
        ),
        LogTransform(),
        ZScoreNormalize(0.0, 1.0),
    ),
    fit=fit,
)
