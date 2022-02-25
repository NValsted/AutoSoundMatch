from typing import Optional
from enum import Enum, unique

from pydantic import BaseModel

from src.flow_synthesizer.base import (
    AEBaseModelEnum,
    EDLayerEnum,
    ModelEnum,
    FlowTypeEnum,
    RegressorEnum,
    LossEnum,
    DisentanglingModelEnum
)


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
    in_dim: list[int]
    out_dim: list[int]


class FlowSynthSection(BaseModel):
    encoding_dim: int
    latent_dim: int
    channels: int
    hidden_dim: int
    ae_base: AEBaseModelEnum
    ed_layer: EDLayerEnum
    model: ModelEnum
    flow_type: Optional[FlowTypeEnum]
    flow_length: Optional[int]
    kernel: int
    dilation: int
    regressor: RegressorEnum
    regressor_flow_type: Optional[FlowTypeEnum]
    regressor_hidden_dim: int
    regressor_layers: int
    reconstruction_loss: LossEnum
    disentangling_model: Optional[DisentanglingModelEnum]
    disentangling_layers: int
    semantic_dim: int = -1


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
