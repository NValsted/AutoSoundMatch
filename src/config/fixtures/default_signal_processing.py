from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from src.config.registry_sections import SignalProcessingSection
from src.daw.signal_transformers import MinMax, StereoToMono

signal_processing_section = SignalProcessingSection(
    pipeline=(
        StereoToMono((None, 2)),
        MinMax(),
        MelSpectrogram(
            sample_rate=22050,
            n_fft=2048,
            n_mels=128,
            hop_length=1024,
            f_min=30,
            f_max=11000,
            pad=0,
        ),
        AmplitudeToDB(top_db=80),
    ),
    fit=None,
)
