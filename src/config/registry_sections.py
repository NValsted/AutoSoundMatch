from enum import Enum, unique
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.config.paths import get_project_root, path_field_factory
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


class PathSection(BaseModel):
    project_root: Path = Field(default_factory=get_project_root)
    downloads: Path = path_field_factory(get_project_root() / "data" / "downloads")
    presets: Path = path_field_factory(get_project_root() / "data" / "presets")
    model: Path = path_field_factory(get_project_root() / "data" / "model")
    audio: Path = path_field_factory(get_project_root() / "data" / "audio")
    midi: Path = path_field_factory(get_project_root() / "data" / "midi")
    processed_audio: Path = path_field_factory(
        get_project_root() / "data" / "processed_audio"
    )


class DatabaseSection(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class SynthSection(BaseModel):
    synth_path: Path
    sample_rate: int = 22050
    buffer_size: int = 128
    bpm: int = 128
    duration: float = 4.0


class DatasetSection(BaseModel):
    in_dim: list[int]
    out_dim: int


class FlowSynthSection(BaseModel):
    """
    Hyperparameters for the flow synthesizer model. The default values
    correspond to the Flow_reg architecture, which is reported to have
    the best audio reconstruction performance in Esling, Philippe, et
    al. (2019).
    """

    encoding_dim: int = 64
    latent_dim: int = 8
    channels: int = 32
    hidden_dim: int = 512
    ae_base: AEBaseModelEnum = AEBaseModelEnum.VAEFlow
    ed_layer: EDLayerEnum = EDLayerEnum.gated_mlp  # TODO : cnn
    model: ModelEnum = ModelEnum.RegressionAE
    n_layers: int = 4
    kernel: int = 5
    dilation: int = 3
    flow_type: Optional[FlowTypeEnum] = FlowTypeEnum.iaf  # TODO: Group parameters
    flow_length: Optional[int] = 16
    regressor: Optional[RegressorEnum] = RegressorEnum.mlp
    regressor_flow_type: Optional[FlowTypeEnum] = FlowTypeEnum.maf
    regressor_hidden_dim: Optional[int] = 256
    regressor_layers: Optional[int] = 3
    reconstruction_loss: Optional[LossEnum] = LossEnum.mse
    disentangling_model: Optional[DisentanglingModelEnum] = None
    disentangling_layers: Optional[int] = None
    semantic_dim: int = -1
    active_model_path: Optional[Path] = None


class TrainMetadataSection(BaseModel):
    epochs: int = 20
    batch_size: int = 64
    time_limit: Optional[int] = 24 * 60
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
    PATH = "PATH"
    DATABASE = "DATABASE"
    SYNTH = "SYNTH"
    DATASET = "DATASET"
    FLOWSYNTH = "FLOWSYNTH"
    TRAINMETA = "TRAINMETA"


RegistrySectionsMap = dict(
    PATH=PathSection,
    DATABASE=DatabaseSection,
    SYNTH=SynthSection,
    DATASET=DatasetSection,
    FLOWSYNTH=FlowSynthSection,
    TRAINMETA=TrainMetadataSection,
)
