from src.config.registry_sections import FlowSynthSection
from src.flow_synthesizer.enums import (
    AEBaseModelEnum,
    EDLayerEnum,
    FlowTypeEnum,
    LossEnum,
    ModelEnum,
    RegressorEnum,
)

flowsynth_config = FlowSynthSection(
    encoding_dim=64,
    latent_dim=32,
    channels=128,
    hidden_dim=2048,
    ae_base=AEBaseModelEnum.WAE,
    ed_layer=EDLayerEnum.cnn,
    model=ModelEnum.RegressionAE,
    n_layers=4,
    kernel=7,
    dilation=3,
    regressor=RegressorEnum.mlp,
    regressor_flow_type=FlowTypeEnum.iaf,
    regressor_hidden_dim=1024,
    regressor_layers=3,
    reconstruction_loss=LossEnum.mse,
)
