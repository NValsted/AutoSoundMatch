import inspect
from dataclasses import dataclass

from src.utils.temporary_context import temporary_attrs
from src.utils.meta import AttributeWrapper
from src.config.base import REGISTRY
from src.config.registry_sections import DatasetSection, FlowSynthSection
from src.flow_synthesizer.base import ModelWrapper, ModelEnum, FlowTypeEnum
from src.flow_synthesizer.acids_ircam_flow_synthesizer.code.models.basic import construct_encoder_decoder, construct_flow


@dataclass
class ModelFactory:
    dim: list[int]
    encoding_dim: int
    latent_dim: int
    model: ModelEnum
    flow_type: FlowTypeEnum
    flow_length: int
    kernel: int
    dilation: int

    def __call__(self, *args, **kwargs) -> ModelWrapper:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            encoder, decoder = construct_encoder_decoder(
                in_size=tmp.dim,
                enc_size=tmp.encoding_dim,
                latent_size=tmp.latent_dim,
                args=AttributeWrapper(
                    kernel=tmp.kernel,
                    dilation=tmp.dilation,
                ),
            )
            model_kwargs = dict(
                encoder=encoder,
                decoder=decoder,
                input_dims=tmp.dim,
                encoder_dims=tmp.encoding_dim,
                latent_dims=tmp.latent_dim,
            )

            model_constructor = tmp.model.value
            if "flow" in inspect.signature(model_constructor).parameters:
                flow, _ = construct_flow(
                    flow_dim=tmp.latent_dim,
                    flow_type=tmp.flow_type.value,
                    flow_length=tmp.flow_length,
                )
                model_kwargs["flow"] = flow
            
            model = model_constructor(**model_kwargs)
            return ModelWrapper(model=model, **kwargs)

    def register(self, commit: bool = False) -> None:
        raise NotImplementedError
        REGISTRY.DATASET = DatasetSection()
        REGISTRY.FLOWSYNTH = FlowSynthSection()
        if commit:
            REGISTRY.commit()
