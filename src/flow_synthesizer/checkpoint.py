from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel

from src.config.registry_sections import FlowSynthSection, TrainMetadataSection
from src.utils.meta import hash_field_to_uuid


class FlowSynthParamsTable(SQLModel, table=True):
    # TODO : This would be more DRY if registry_sections were SQLModel classes
    __tablename__ = "FlowSynthParams"

    id: Optional[str] = Field(primary_key=True, default=None)
    encoding_dim: int
    latent_dim: int
    channels: int
    hidden_dim: int
    ae_base: str
    ed_layer: str
    model: str
    n_layers: int
    kernel: int
    dilation: int
    flow_type: Optional[str]
    flow_length: Optional[int]
    regressor: Optional[str]
    regressor_flow_type: Optional[str]
    regressor_hidden_dim: Optional[int]
    regressor_layers: Optional[int]
    reconstruction_loss: Optional[str]
    disentangling_model: Optional[str]
    disentangling_layers: Optional[int]
    semantic_dim: int

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")

    @classmethod
    def from_registry_section(
        cls, flow_synth_section: FlowSynthSection
    ) -> "FlowSynthParamsTable":
        return cls(
            encoding_dim=flow_synth_section.encoding_dim,
            latent_dim=flow_synth_section.latent_dim,
            channels=flow_synth_section.channels,
            hidden_dim=flow_synth_section.hidden_dim,
            ae_base=str(flow_synth_section.ae_base),
            ed_layer=str(flow_synth_section.ed_layer),
            model=str(flow_synth_section.model),
            n_layers=flow_synth_section.n_layers,
            kernel=flow_synth_section.kernel,
            dilation=flow_synth_section.dilation,
            flow_type=str(flow_synth_section.flow_type),
            flow_length=flow_synth_section.flow_length,
            regressor=str(flow_synth_section.regressor),
            regressor_flow_type=str(flow_synth_section.regressor_flow_type),
            regressor_hidden_dim=flow_synth_section.regressor_hidden_dim,
            regressor_layers=flow_synth_section.regressor_layers,
            reconstruction_loss=str(flow_synth_section.reconstruction_loss),
            disentangling_model=str(flow_synth_section.disentangling_model),
            disentangling_layers=flow_synth_section.disentangling_layers,
            semantic_dim=flow_synth_section.semantic_dim,
        )


class TrainMetadataParamsTable(SQLModel, table=True):
    __tablename__ = "TrainMetadataParams"

    id: Optional[str] = Field(primary_key=True, default=None)
    in_dim: str
    out_dim: int
    epochs: int
    batch_size: int
    time_limit: Optional[int]
    loss: str
    learning_rate: float
    scheduler_mode: str
    scheduler_factor: float
    scheduler_patience: int
    scheduler_verbose: bool
    scheduler_threshold: float
    beta_factor: float
    reg_factor: float
    start_regress: int
    warm_regress: int
    warm_latent: int

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")

    @classmethod
    def from_registry_section(
        cls, train_metadata_section: TrainMetadataSection
    ) -> "TrainMetadataParamsTable":
        return cls(
            in_dim=str(train_metadata_section.in_dim),
            out_dim=train_metadata_section.out_dim,
            epochs=train_metadata_section.epochs,
            batch_size=train_metadata_section.batch_size,
            time_limit=train_metadata_section.time_limit,
            loss=str(train_metadata_section.loss),
            learning_rate=train_metadata_section.learning_rate,
            scheduler_mode=str(train_metadata_section.scheduler_mode),
            scheduler_factor=train_metadata_section.scheduler_factor,
            scheduler_patience=train_metadata_section.scheduler_patience,
            scheduler_verbose=train_metadata_section.scheduler_verbose,
            scheduler_threshold=train_metadata_section.scheduler_threshold,
            beta_factor=train_metadata_section.beta_factor,
            reg_factor=train_metadata_section.reg_factor,
            start_regress=train_metadata_section.start_regress,
            warm_regress=train_metadata_section.warm_regress,
            warm_latent=train_metadata_section.warm_latent,
        )


class ModelCheckpointTable(SQLModel, table=True):
    __tablename__ = "ModelCheckpoint"

    model_id: str = Field(primary_key=True)
    checkpoint_path: Optional[str] = Field()
    accumulated_epochs: Optional[int] = Field()
    val_loss: Optional[float] = Field()
    flow_synth_params: Optional[str] = Field(foreign_key="FlowSynthParams.id")
    train_metadata_params: Optional[str] = Field(foreign_key="TrainMetadataParams.id")
    dataset_params: Optional[str] = Field(foreign_key="DatasetParams.id")
    time: datetime = Field(primary_key=True, default_factory=datetime.utcnow)
