from typing import Optional
from enum import Enum, unique

from pydantic import BaseModel

from src.flow_synthesizer.base import ModelEnum, FlowTypeEnum


class DatabaseSection(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class SynthSection(BaseModel):
    synth_path: str
    sample_rate: int = 44100
    buffer_size: int = 128
    bpm: int = 128
    duration: float = 1.0


class DatasetSection(BaseModel):
    dim: list[int]


class FlowSynthSection(BaseModel):
    encoding_dim: int
    latent_dim: int
    model: ModelEnum
    flow_type: FlowTypeEnum
    flow_length: int
    kernel: int
    dilation: int


@unique
class RegistrySectionsEnum(str, Enum):
    DATABASE = "DATABASE"
    SYNTH = "SYNTH"
    DATASET = "DATASET"
    FLOWSYNTH = "FLOWSYNTH"


RegistrySectionsMap = dict(
    DATABASE=DatabaseSection,
    SYNTH=SynthSection,
    DATASET=DatasetSection,
    FLOWSYNTH=FlowSynthSection,
)
