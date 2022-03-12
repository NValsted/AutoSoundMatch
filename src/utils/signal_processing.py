import librosa
import numpy as np


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
