from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from src.config.base import REGISTRY
from src.config.registry_sections import (
    DatasetSection,
    FlowSynthSection,
    TrainMetadataSection,
)
from src.daw.audio_model import AudioBridgeTable
from src.flow_synthesizer.base import ModelWrapper
from src.flow_synthesizer.factory import ModelFactory


@dataclass
class ModelSuite:  # TODO : Ensure each model is only loaded into memory when needed
    MLP: ModelWrapper
    CNN: ModelWrapper
    ResNet: ModelWrapper
    AE: ModelWrapper
    VAE: ModelWrapper
    WAE: ModelWrapper
    VAE_flow: ModelWrapper
    Flow_reg: ModelWrapper
    Flow_dis: ModelWrapper

    def __post_init__(self):
        # TODO : remember to remove when implementing get_model_suite
        raise NotImplementedError


def prepare_registry(dataset: Optional[Dataset] = None, commit: bool = False) -> None:
    """
    Prepare the registry with default values for the model training and evaluation.
    """
    if dataset is not None:
        REGISTRY.DATASET = DatasetSection(
            in_dim=dataset.in_dim,
            out_dim=dataset.out_dim,
        )

    if REGISTRY.TRAINMETA is None:
        REGISTRY.TRAINMETA = TrainMetadataSection()

    if REGISTRY.FLOWSYNTH is None:
        REGISTRY.FLOWSYNTH = FlowSynthSection()

    if commit:
        REGISTRY.commit()


def get_model(
    hyper_parameters: Optional[FlowSynthSection] = None,
    dataset: Optional[Dataset] = None,
) -> ModelWrapper:

    if hyper_parameters is None:
        if REGISTRY.FLOWSYNTH is None:
            raise ValueError("REGISTRY.FLOWSYNTH section is not set.")
        hyper_parameters = REGISTRY.FLOWSYNTH

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
        encoding_dim=hyper_parameters.encoding_dim,
        latent_dim=hyper_parameters.latent_dim,
        channels=hyper_parameters.channels,
        hidden_dim=hyper_parameters.hidden_dim,
        ae_base=hyper_parameters.ae_base,
        ed_layer=hyper_parameters.ed_layer,
        model=hyper_parameters.model,
        flow_type=hyper_parameters.flow_type,
        flow_length=hyper_parameters.flow_length,
        n_layers=hyper_parameters.n_layers,
        kernel=hyper_parameters.kernel,
        dilation=hyper_parameters.dilation,
        regressor=hyper_parameters.regressor,
        regressor_flow_type=hyper_parameters.regressor_flow_type,
        regressor_hidden_dim=hyper_parameters.regressor_hidden_dim,
        regressor_layers=hyper_parameters.regressor_layers,
        reconstruction_loss=hyper_parameters.reconstruction_loss,
        disentangling_model=hyper_parameters.disentangling_model,
        disentangling_layers=hyper_parameters.disentangling_layers,
        semantic_dim=-hyper_parameters.semantic_dim,
    )
    model = model_factory()
    return model


def get_model_suite():
    raise NotImplementedError


def evaluate_inference(model: ModelWrapper, audio_bridges: list[AudioBridgeTable]):
    raise NotImplementedError
