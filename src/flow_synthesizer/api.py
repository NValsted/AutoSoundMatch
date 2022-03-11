from typing import Optional

from torch.utils.data import Dataset, DataLoader

from src.config.base import REGISTRY
from src.config.registry_sections import TrainMetadataSection
from src.flow_synthesizer.factory import ModelFactory
from src.flow_synthesizer.enums import (
    AEBaseModelEnum,
    EDLayerEnum,
    ModelEnum,
    FlowTypeEnum,
    RegressorEnum,
    # DisentanglingModelEnum,
    LossEnum,
)
from src.flow_synthesizer.base import ModelWrapper
from src.database.dataset import PolyDataset
from src.database.factory import DBFactory


def prepare_registry(dataset: Optional[Dataset] = None) -> None:
    """
    Prepare the registry for the model training and evaluation.
    """
    if dataset is not None:
        raise NotImplementedError
    REGISTRY.TRAINMETA = TrainMetadataSection()


def get_flow_reg_model(dataset: Optional[Dataset] = None) -> ModelWrapper:
    """
    Get a model with the Flow_reg architecture, which is reported to
    have the best audio reconstruction performance in Esling, Philippe,
    et al. (2019).
    """

    if dataset is not None:
        in_dim = dataset.in_dim
        out_dim = dataset.out_dim
    else:
        if REGISTRY.DATASET is None:
            raise ValueError("REGISTRY.DATASET section is not set.")
        in_dim = REGISTRY.DATASET.in_dim
        out_dim = REGISTRY.DATASET.out_dim

    model_factory = ModelFactory(
        in_dim=in_dim,
        out_dim=out_dim,
        encoding_dim=64,
        latent_dim=8,
        channels=32,
        hidden_dim=512,
        ae_base=AEBaseModelEnum.VAEFlow,
        ed_layer=EDLayerEnum.gated_mlp,
        model=ModelEnum.RegressionAE,
        flow_type=FlowTypeEnum.iaf,
        flow_length=16,
        n_layers=4,
        kernel=5,
        dilation=3,
        regressor=RegressorEnum.mlp,
        regressor_flow_type=FlowTypeEnum.maf,
        regressor_hidden_dim=256,
        regressor_layers=3,
        reconstruction_loss=LossEnum.mse,
        # disentangling_model=DisentanglingModelEnum.density,
        # disentangling_layers=8,
        # semantic_dim=-1,
    )
    model = model_factory()
    return model


def get_model_suite():
    raise NotImplementedError
