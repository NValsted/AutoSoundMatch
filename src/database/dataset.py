from typing import Tuple

import librosa
import numpy as np
import torch
from sqlmodel import select
from torch.utils.data import Dataset

from src.database.base import Database
from src.daw.audio_model import AudioBridgeTable
from src.daw.synth_model import SynthParamsTable


class PolyDataset(Dataset):
    db: Database
    audio_bridges: list[AudioBridgeTable]
    in_dim: list[int, int]
    out_dim: int

    def __init__(self, db: Database, test_flag: bool = False):
        self.db = db

        with self.db.session() as session:
            query = select(AudioBridgeTable).where(
                AudioBridgeTable.test_flag == test_flag
            )
            self.audio_bridges = session.exec(query).all()

        if len(self.audio_bridges) == 0:
            raise ValueError(
                f"No audio bridges found in database {db.engine} with {test_flag=}"
            )

        sample = self.__getitem__(0)
        self.in_dim = sample[0].shape
        self.out_dim = sample[1].shape[0]

    def _process_signal(self, signal: np.ndarray, sample_rate: int):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_fft=2048,
            n_mels=128,
            hop_length=1024,
            fmin=30,
            fmax=11000,
        )
        return librosa.power_to_db(mel_spectrogram)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        with self.db.session() as session:
            audio_bridge = self.audio_bridges[index]

            query = select(SynthParamsTable.__table__).where(
                SynthParamsTable.id == audio_bridge.synth_params
            )

            synth_params = session.execute(query).first()
            signal, sample_rate = librosa.load(audio_bridge.audio_path)
            processed_signal = self._process_signal(signal, sample_rate)

            return (
                torch.from_numpy(processed_signal).float(),
                torch.tensor(synth_params[1:]).float(),
                [],
                signal,
            )

    def __len__(self):
        return len(self.audio_bridges)
