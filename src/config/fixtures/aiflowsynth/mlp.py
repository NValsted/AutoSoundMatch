from src.config.registry_sections import FlowSynthSection
from src.flow_synthesizer.enums import ModelEnum

flowsynth_config = FlowSynthSection(
    hidden_dim=2048,
    model=ModelEnum.MLP,
    n_layers=4,
)
