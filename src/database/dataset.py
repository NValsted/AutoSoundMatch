import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from sqlmodel import Field, SQLModel, select
from sqlmodel.sql.expression import SelectOfScalar
from torch.utils.data import IterableDataset

from src.config.base import PYTORCH_DEVICE, REGISTRY
from src.config.registry_sections import DatasetSection
from src.database.base import Database
from src.daw.audio_model import AudioBridgeTable
from src.daw.signal_processing import SIGNAL_PROCESSOR
from src.daw.synth_model import SynthParamsTable
from src.utils.meta import hash_field_to_uuid
from src.utils.temporary_context import temporary_attrs

SelectOfScalar.inherit_cache = True


@dataclass
class FlowSynthDataset(IterableDataset):
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
                f"No audio bridges found in database {self.db.engine.url} with"
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
                    if audio_bridge.processed_path is None:  # TODO : concurrent load
                        signal = torch.load(audio_bridge.audio_path).to(PYTORCH_DEVICE)
                        processed_signal = SIGNAL_PROCESSOR(signal)
                    else:
                        processed_signal = torch.load(audio_bridge.processed_path).to(
                            PYTORCH_DEVICE
                        )

                    self._cache.append(
                        (
                            processed_signal.float(),
                            torch.Tensor(
                                synth_params_dict[audio_bridge.synth_params]
                            ).float(),
                            [],
                            [],
                        )
                    )

        self._expected_index += 1
        return self._cache.pop()

    def __len__(self):
        return len(self.audio_bridges)


def load_formatted_audio(audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads an audio file, and prepares it to be used in inference step.
    """

    if re.search(r"\.wav$", audio_path):
        signal, sample_rate = torchaudio.load(audio_path)
        if sample_rate != REGISTRY.SYNTH.sample_rate:
            resample_transform = torchaudio.transforms.Resample(
                sample_rate, REGISTRY.SYNTH.sample_rate
            )
            signal = resample_transform(signal)

    elif re.search(r"\.pt$", audio_path):
        signal = torch.load(audio_path).to(PYTORCH_DEVICE)

    elif re.search(r"\.npy$", audio_path):
        signal_as_ndarray = np.load(audio_path)
        signal = torch.from_numpy(signal_as_ndarray).float().to(PYTORCH_DEVICE)

    else:
        raise ValueError(f"Unsupported file format: {audio_path}")

    processed = SIGNAL_PROCESSOR(signal)
    formatted = processed.reshape(1, *processed.shape).contiguous().to(PYTORCH_DEVICE)
    return formatted, signal


class DatasetParamsTable(SQLModel, table=True):
    __tablename__ = "DatasetParams"

    id: Optional[str] = Field(primary_key=True, default=None)
    num_presets: int = Field()
    num_midi: int = Field()
    pairs: int = Field()
    time: datetime = Field(primary_key=True, default_factory=datetime.utcnow)

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")

    @classmethod
    def from_registry_section(
        cls, dataset_section: DatasetSection
    ) -> "DatasetParamsTable":
        return cls(
            num_presets=dataset_section.num_presets,
            num_midi=dataset_section.num_midi,
            pairs=dataset_section.pairs,
        )
