from dataclasses import dataclass, field
from typing import Tuple

import librosa
import numpy as np
import torch
from sqlmodel import select
from torch.utils.data import IterableDataset

from src.database.base import Database
from src.daw.audio_model import AudioBridgeTable
from src.daw.synth_model import SynthParamsTable
from src.utils.signal_processing import process_sample
from src.utils.temporary_context import temporary_attrs


@dataclass
class PolyDataset(IterableDataset):
    db: Database
    test_flag: bool = False
    shuffle: bool = True
    cache_size: int = 4096
    audio_bridges: list[AudioBridgeTable] = field(init=False)
    in_dim: list[int, int] = field(init=False)
    out_dim: int = field(init=False)
    _cache: list = field(default_factory=list)
    _expected_index: int = 0

    def __post_init__(self):
        with self.db.session() as session:
            query = select(AudioBridgeTable).where(
                AudioBridgeTable.test_flag == self.test_flag
            )
            self.audio_bridges = session.exec(query).all()

        if len(self.audio_bridges) == 0:
            raise ValueError(
                f"No audio bridges found in database {self.db.engine} with"
                f" {self.test_flag=}"
            )

        sample = self[0]
        self.in_dim = sample[0].shape
        self.out_dim = sample[1].shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        with temporary_attrs(
            self, _cache=[], _expected_index=index, cache_size=1
        ) as tmp:
            return next(tmp)

    def __iter__(self):
        self._expected_index = 0
        if self.shuffle:
            np.random.shuffle(self.audio_bridges)
        return self

    def __next__(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        if len(self._cache) == 0:
            with self.db.session() as session:

                if (self._expected_index + self.cache_size) >= len(self.audio_bridges):
                    num_to_retrieve = len(self.audio_bridges) - self._expected_index - 1
                else:
                    num_to_retrieve = self.cache_size

                audio_bridges = [
                    self.audio_bridges[self._expected_index + i]
                    for i in range(num_to_retrieve - 1, -1, -1)
                ]  # reverse order to allow pop operation when retrieving from cache
                if len(audio_bridges) == 0:
                    raise StopIteration

                query = select(SynthParamsTable.__table__).filter(
                    SynthParamsTable.id.in_(
                        [bridge.synth_params for bridge in audio_bridges]
                    )
                )

                synth_params_dict = {
                    params[0]: params[1:] for params in session.execute(query).all()
                }

                for audio_bridge in audio_bridges:
                    signal, sample_rate = librosa.load(audio_bridge.audio_path)
                    processed_signal = process_sample(signal, sample_rate)
                    self._cache.append(
                        self._format_output(
                            processed_signal,
                            synth_params_dict[audio_bridge.synth_params],
                            [],
                            signal,
                        )
                    )

        self._expected_index += 1
        return self._cache.pop()

    @staticmethod
    def _format_output(
        processed_signal: np.ndarray,
        synth_params: np.ndarray,
        metadata: list,
        signal: np.ndarray,
    ):
        return (
            torch.from_numpy(processed_signal).float(),
            torch.tensor(synth_params).float(),
            metadata,
            signal,
        )

    def __len__(self):
        return len(self.audio_bridges)
