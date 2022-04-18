import inspect
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Optional, Union

from torch import nn

from src.config.base import PYTORCH_DEVICE, REGISTRY
from src.config.registry_sections import DatasetSection, FlowSynthSection
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import (
    DecodeCNN,
    DecodeMLP,
    GatedCNN,
    GatedMLP,
    construct_disentangle,
    construct_encoder_decoder,
    construct_flow,
    construct_regressor,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.disentangling import (  # NOQA: E501
    DisentanglingFlow,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.flows.flow import (
    NormalizingFlow,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.loss import (
    multinomial_loss,
    multinomial_mse_loss,
)
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.regression import (
    BayesianRegressor,
    FlowPredictor,
)
from src.flow_synthesizer.base import ModelWrapper
from src.flow_synthesizer.enums import (
    AEBaseModelEnum,
    DisentanglingModelEnum,
    EDLayerEnum,
    FlowTypeEnum,
    LossEnum,
    ModelEnum,
    ModelEnumConstructorMap,
    RegressorEnum,
)
from src.utils.meta import AttributeWrapper
from src.utils.temporary_context import temporary_attrs


@dataclass
class ModelFactory:
    in_dim: list[int]
    out_dim: int
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

    def __call__(self, *args, **kwargs) -> ModelWrapper:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            tmp: ModelFactory

            if tmp.model in (ModelEnum.MLP, ModelEnum.GatedMLP):
                model = tmp._MLP()
                return ModelWrapper(model=model)

            elif tmp.model in (ModelEnum.CNN, ModelEnum.ResCNN, ModelEnum.GatedCNN):
                model = tmp._CNN()
                return ModelWrapper(model=model)

            else:
                encoder, decoder = tmp._encoder_decoder()

                ae_base_kwargs = dict(
                    encoder=encoder,
                    decoder=decoder,
                    encoder_dims=tmp.encoding_dim,
                    latent_dims=tmp.latent_dim,
                )

                ae_base_constructor = tmp.ae_base.value
                if "flow" in inspect.signature(ae_base_constructor).parameters:
                    flow = tmp._flow()
                    ae_base_kwargs["flow"] = flow
                if "input_dims" in inspect.signature(ae_base_constructor).parameters:
                    ae_base_kwargs["input_dims"] = tmp.in_dim

                ae_base = ae_base_constructor(**ae_base_kwargs)
                regressor = tmp._regressor()
                reconstruction_loss = tmp._reconstruction_loss()

                if tmp.model == ModelEnum.RegressionAE:
                    constructor = ModelEnumConstructorMap[tmp.model.value]
                    model = constructor(
                        ae_model=ae_base,
                        latent_dims=tmp.latent_dim,
                        regression_dims=tmp.out_dim,
                        recons_loss=reconstruction_loss,
                        regressor=regressor,
                        regressor_name=self.regressor.value,
                    )

                else:
                    if tmp.semantic_dim <= 0:
                        raise ValueError("semantic_dim must be > 0")
                    disentangling = tmp._disentangling()

                    constructor = ModelEnumConstructorMap[tmp.model.value]
                    model = constructor(
                        ae_model=ae_base,
                        latent_dims=tmp.latent_dim,
                        regression_dims=tmp.out_dim,
                        recons_loss=reconstruction_loss,
                        regressor=regressor,
                        regressor_name=self.regressor.value,
                        disentangling=disentangling,
                        semantic_dim=tmp.semantic_dim,
                    )

                return ModelWrapper(model=model)

    def _MLP(self) -> GatedMLP:
        constructor = ModelEnumConstructorMap[self.model.value]
        return constructor(
            in_size=reduce(lambda a, b: a * b, self.in_dim),
            out_size=self.out_dim,
            hidden_size=self.hidden_dim,
            n_layers=self.n_layers,
        )

    def _CNN(self) -> GatedCNN:
        constructor = ModelEnumConstructorMap[self.model.value]
        return constructor(
            in_size=self.in_dim,
            out_size=self.out_dim,
            channels=self.channels,
            n_layers=4,  # TODO : static in train.py but might parametrize here
            hidden_size=self.hidden_dim,
            n_mlp=3,  # TODO : static in train.py but might parametrize here,
            args=AttributeWrapper(  # TODO : dropout is also static
                kernel=self.kernel,
                dilation=self.dilation,
            ),
        )

    def _encoder_decoder(
        self,
    ) -> tuple[Union[GatedMLP, GatedCNN], Union[DecodeMLP, DecodeCNN]]:
        encoder, decoder = construct_encoder_decoder(
            in_size=self.in_dim,
            enc_size=self.encoding_dim,
            latent_size=self.latent_dim,
            channels=self.channels,
            n_layers=self.n_layers,
            hidden_size=self.hidden_dim,
            n_mlp=self.n_layers // 2,
            type_mod=self.ed_layer.value,
            args=AttributeWrapper(
                kernel=self.kernel,
                dilation=self.dilation,
            ),
        )
        if isinstance(decoder, DecodeCNN):
            last_layer = decoder.net[-1]
            constructor = type(last_layer)
            decoder.net[-1] = constructor(
                last_layer.in_channels,
                1,
                last_layer.kernel_size,
                last_layer.stride,
                last_layer.padding,
                output_padding=last_layer.output_padding,
                dilation=last_layer.dilation,
            )

        return encoder, decoder

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

    def _regressor(self) -> Union[FlowPredictor, BayesianRegressor, nn.Sequential]:
        if self.regressor_flow_type is None and self.regressor != RegressorEnum.mlp:
            raise ValueError("regressor_flow_type is required with current parameters")
        if self.regressor is None:
            raise ValueError("regressor is required with current parameters")
        if self.regressor_hidden_dim is None and self.regressor in (
            RegressorEnum.mlp,
            RegressorEnum.bnn,
        ):
            raise ValueError("regressor_hidden_dim is required with current parameters")
        return construct_regressor(
            in_dims=self.latent_dim,
            out_dims=self.out_dim,
            model=self.regressor.value,
            hidden_dims=self.regressor_hidden_dim,
            n_layers=self.regressor_layers,
            flow_type=self.regressor_flow_type.value
            if self.regressor_flow_type
            else None,
        )

    def _reconstruction_loss(self) -> Callable:
        if self.reconstruction_loss is None:
            raise ValueError("reconstruction_loss is required with current parameters")
        if self.reconstruction_loss == LossEnum.mse:
            return nn.MSELoss(reduction="sum").to(PYTORCH_DEVICE)
        elif self.reconstruction_loss == LossEnum.l1:
            return nn.SmoothL1Loss(reduction="sum").to(PYTORCH_DEVICE)
        elif self.reconstruction_loss == LossEnum.multinomial:
            return multinomial_loss
        elif self.reconstruction_loss == LossEnum.multi_mse:
            return multinomial_mse_loss
        raise ValueError(f"{self.reconstruction_loss=} is invalid")

    def _disentangling(self) -> DisentanglingFlow:
        if self.disentangling_model is None:
            raise ValueError("disentangling_model is required with current parameters")
        if self.disentangling_layers is None:
            raise ValueError("disentangling_layers is required with current parameters")
        return construct_disentangle(
            in_dims=self.latent_dim,
            model=self.disentangling_model.value,
            n_layers=self.disentangling_layers,
            flow_type=self.regressor_flow_type.value,
        )

    def register(self, commit: bool = False) -> None:
        raise NotImplementedError
        REGISTRY.DATASET = DatasetSection()
        REGISTRY.FLOWSYNTH = FlowSynthSection()
        if commit:
            REGISTRY.commit()
