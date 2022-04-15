from dataclasses import dataclass
from pathlib import Path
from typing import Union

from dawdreamer import PluginProcessor, RenderEngine

from src.config.base import REGISTRY
from src.config.registry_sections import SynthSection
from src.daw.base import SynthHost
from src.utils.temporary_context import temporary_attrs


@dataclass
class SynthHostFactory:
    synth_path: Path
    sample_rate: int
    buffer_size: int
    duration: float
    bpm: int
    locked_parameters: dict[Union[str, int], float]

    def __call__(self, *args, **kwargs) -> SynthHost:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            tmp: SynthHostFactory

            engine = RenderEngine(tmp.sample_rate, tmp.buffer_size)
            engine.set_bpm(tmp.bpm)
            synth = engine.make_plugin_processor("synth", str(tmp.synth_path))
            engine.load_graph([(synth, [])])
            normalized_locked_parameters = tmp._normalize_parameters(
                synth, tmp.locked_parameters
            )

            return SynthHost(
                engine=engine,
                synth=synth,
                locked_parameters=normalized_locked_parameters,
                **kwargs,
            )

    @staticmethod
    def _normalize_parameters(synth: PluginProcessor, parameters: dict) -> dict:
        normalized_parameters = {}
        param_name_map = {
            param["name"]: int(param["index"])
            for param in synth.get_plugin_parameters_description()
        }

        for k, v in parameters.items():
            if isinstance(k, str):
                if k.isdigit():
                    normalized_parameters[int(k)] = v
                elif k in param_name_map:
                    normalized_parameters[param_name_map[k]] = v
                else:
                    raise ValueError(
                        f"Invalid parameter name: {k} - possible options are"
                        f" {param_name_map.keys()}"
                    )

            elif isinstance(k, int):
                normalized_parameters[int(k)] = v

            else:
                raise ValueError(
                    f"Invalid parameter: {k} - possible options are"
                    f" {param_name_map.keys()}"
                )

        return normalized_parameters

    def register(self, commit: bool = False) -> None:
        REGISTRY.SYNTH = SynthSection(
            synth_path=self.synth_path,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            bpm=self.bpm,
        )
        if commit:
            REGISTRY.commit()
