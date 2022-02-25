import inspect
from typing import Callable, Optional, Union
from dataclasses import dataclass

from torch import nn

from src.utils.temporary_context import temporary_attrs
from src.utils.meta import AttributeWrapper
from src.config.base import REGISTRY
from src.config.registry_sections import DatasetSection, FlowSynthSection
from src.flow_synthesizer.base import (
    ModelWrapper,
    AEBaseModelEnum,
    ModelEnum,
    FlowTypeEnum,
    RegressorEnum,
    LossEnum,
    DisentanglingModelEnum
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.loss import multinomial_loss, multinomial_mse_loss
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.flows.flow import NormalizingFlow
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.disentangling import DisentanglingFlow
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import (
    construct_encoder_decoder,
    construct_flow,
    construct_regressor,
    construct_disentangle,
    GatedMLP,
    GatedCNN,
    DecodeMLP,
    DecodeCNN,
)


@dataclass
class ModelFactory:
    in_dim: list[int]
    out_dim: list[int]
    encoding_dim: int
    latent_dim: int
    ae_base: AEBaseModelEnum
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

    def __call__(self, *args, **kwargs) -> ModelWrapper:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            if tmp.model == ModelEnum.GatedMLP:
                raise NotImplementedError
                return ModelWrapper(model=model)

            elif tmp.model == ModelEnum.GatedCNN:
                raise NotImplementedError
                return ModelWrapper(model=model)

            else:
                encoder, decoder = tmp._encoder_decoder()

                ae_base_kwargs = dict(
                    encoder=encoder,
                    decoder=decoder,
                    input_dims=tmp.in_dim,
                    encoder_dims=tmp.encoding_dim,
                    latent_dims=tmp.latent_dim,
                )

                ae_base_constructor = tmp.ae_base.value
                if "flow" in inspect.signature(ae_base_constructor).parameters:
                    flow = tmp._flow()
                    ae_base_kwargs["flow"] = flow

                ae_base = ae_base_constructor(**ae_base_kwargs)
                regressor = tmp._regressor()
                reconstruction_loss = tmp._reconstruction_loss()

                if tmp.model == ModelEnum.RegressionAE:
                    constructor = ModelEnum.RegressionAE.value
                    model = constructor(
                        ae_model=ae_base,
                        latent_dims=tmp.latent_dim,
                        regression_dims=tmp.out_dim,
                        recons_loss=reconstruction_loss,
                        regressor=regressor,
                    )

                else:
                    if tmp.semantic_dim <= 0:
                        raise ValueError("semantic_dim must be > 0")
                    disentangling = tmp._disentangling()

                    constructor = ModelEnum.DisentanglingAE.value
                    model = constructor(
                        ae_model=ae_base,
                        latent_dims=tmp.latent_dim,
                        regression_dims=tmp.out_dim,
                        recons_loss=reconstruction_loss,
                        regressor=regressor,
                        disentangling=disentangling,
                        semantic_dim=tmp.semantic_dim,
                    )

                return ModelWrapper(model=model)  # TODO: .to(device)

    def _encoder_decoder(self) -> tuple[Union[GatedMLP, GatedCNN], Union[DecodeMLP, DecodeCNN]]:
        return construct_encoder_decoder(
            in_size=self.in_dim,
            enc_size=self.encoding_dim,
            latent_size=self.latent_dim,
            args=AttributeWrapper(
                kernel=self.kernel,
                dilation=self.dilation,
            ),
        )

    def _flow(self) -> NormalizingFlow:
        if self.flow_type is None:
            raise ValueError("flow_type is required with current parameters")
        if self.flow_length is None:
            raise ValueError("flow_length is required with current parameters")
        flow, _ = construct_flow(
            flow_dim=self.latent_dim,
            flow_type=self.flow_type.value,
            flow_length=self.flow_length,
        )
        return flow

    def _regressor(self):
        if self.regressor_flow_type is None and self.regressor != RegressorEnum.MLP:
            raise ValueError("regressor_flow_type is required with current parameters")
        return construct_regressor(
            in_dims=self.latent_dim,
            out_dims=self.out_dim,
            regressor=self.regressor.value,
            hidden_dims=self.regressor_hidden_dim,
            n_layers=self.regressor_layers,
            flow_type=self.regressor_flow_type.value if self.regressor_flow_type else None,
        )

    def _reconstruction_loss(self) -> Callable:
        if self.reconstruction_loss == LossEnum.mse:
            return nn.MSELoss(reduction="sum")  # TODO : .to(device)
        elif self.reconstruction_loss == LossEnum.l1:
            return nn.SmoothL1Loss(reduction="sum")  # TODO : .to(device)
        elif self.reconstruction_loss == LossEnum.multinomial:
            return multinomial_loss
        elif self.reconstruction_loss == LossEnum.multi_mse:
            return multinomial_mse_loss
        raise ValueError(f"{self.reconstruction_loss=} is invalid")

    def _disentangling(self) -> DisentanglingFlow:
        if self.disentangling_model is None:
            raise ValueError("disentangling_model is required with current parameters")
        return construct_disentangle(
            in_dims=self.latent_dim,
            model=self.disentangling_model.value,
            n_layers=self.disentangling_layers,
            flow_type=self.regressor_flow_type.value
        )

    def register(self, commit: bool = False) -> None:
        raise NotImplementedError
        REGISTRY.DATASET = DatasetSection()
        REGISTRY.FLOWSYNTH = FlowSynthSection()
        if commit:
            REGISTRY.commit()
