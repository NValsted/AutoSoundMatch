from enum import Enum, unique
from typing import Optional

from pydantic import BaseModel

from src.flow_synthesizer.enums import (
    AEBaseModelEnum,
    DisentanglingModelEnum,
    EDLayerEnum,
    FlowTypeEnum,
    LossEnum,
    ModelEnum,
    RegressorEnum,
    SchedulerModeEnum,
)


class DatabaseSection(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class SynthSection(BaseModel):
    synth_path: str
    sample_rate: int = 22050
    buffer_size: int = 128
    bpm: int = 128
    duration: float = 4.0


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
    n_layers: int
    kernel: int
    dilation: int
    flow_type: Optional[FlowTypeEnum] = None  # TODO: Group parameters
    flow_length: Optional[int] = None
    regressor: Optional[RegressorEnum] = None
    regressor_flow_type: Optional[FlowTypeEnum] = None
    regressor_hidden_dim: Optional[int] = None
    regressor_layers: Optional[int] = None
    reconstruction_loss: Optional[LossEnum] = None
    disentangling_model: Optional[DisentanglingModelEnum] = None
    disentangling_layers: Optional[int] = None
    semantic_dim: int = -1


class TrainMetadataSection(BaseModel):
    epochs: int = 200
    batch_size: int = 64
    loss: LossEnum = LossEnum.mse
    learning_rate: float = 2e-4
    scheduler_mode: SchedulerModeEnum = SchedulerModeEnum.min
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    scheduler_verbose: bool = True
    scheduler_threshold: float = 1e-7
    beta_factor: float = 1.0
    reg_factor: float = 1e3
    start_regress: int = 1e2
    warm_regress: int = 1e2
    warm_latent: int = 50


@unique
class RegistrySectionsEnum(str, Enum):
    DATABASE = "DATABASE"
    SYNTH = "SYNTH"
    DATASET = "DATASET"
    FLOWSYNTH = "FLOWSYNTH"
    TRAINMETA = "TRAINMETA"


RegistrySectionsMap = dict(
    DATABASE=DatabaseSection,
    SYNTH=SynthSection,
    DATASET=DatasetSection,
    FLOWSYNTH=FlowSynthSection,
    TRAINMETA=TrainMetadataSection,
)
