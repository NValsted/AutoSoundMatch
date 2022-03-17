import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from dawdreamer import PluginProcessor, RenderEngine
from scipy.stats import uniform

from src.daw.render_model import RenderParams
from src.utils.code_generation import INDENT, get_code_gen_header, sanitize_attribute

if TYPE_CHECKING:
    from src.daw.synth_model import SynthParams, SynthParamsTable


@dataclass
class SynthHost:
    engine: RenderEngine
    synth: PluginProcessor
    _synth_model_path: str = (
        f"{os.path.dirname(os.path.realpath(__file__))}/synth_model.py"
    )

    def create_parameter_table(self) -> str:
        sorted_parameters = sorted(
            self.synth.get_plugin_parameters_description(), key=lambda x: x["index"]
        )
        fields = [
            f'{sanitize_attribute(param["name"])}: float ='
            f' Field(alias="{param["index"]}")'
            for param in sorted_parameters
        ]

        assert len(fields) == len(set(fields))

        str_field_names = map(lambda x: '"{}"'.format(x.split(":")[0]), fields)

        with open(self._synth_model_path, "w") as f:
            f.write(get_code_gen_header())
            f.write("from typing import Optional\n\n")
            f.write("from sqlmodel import SQLModel, Field\n")
            f.write("from sqlalchemy import UniqueConstraint\n\n")
            f.write("from src.utils.meta import hash_field_to_uuid\n\n\n")
            f.write("class SynthParams(SQLModel):\n")
            f.write(
                f"{INDENT}id: Optional[str] = Field(primary_key=True,"
                f" default=None)\n{INDENT}"
            )
            f.write(f"\n{INDENT}".join(fields) + "\n\n")
            f.write(f"{INDENT}class Config:\n")
            f.write(f"{INDENT*2}validate_all = True\n\n")
            f.write(f'{INDENT}_auto_uuid = hash_field_to_uuid("id")\n\n\n')
            f.write(f"class SynthParamsTable(SynthParams, table=True):\n{INDENT}")
            f.write(f'__tablename__ = "SynthParams"\n{INDENT}')
            f.write(
                f"__table_args__ = (UniqueConstraint(\n{INDENT*2}"
                + f",\n{INDENT*2}".join(str_field_names)
                + f"\n{INDENT}),)\n"
            )

        return self._synth_model_path

    def get_patch_as_model(
        self, table: bool = False
    ) -> Union["SynthParams", "SynthParamsTable"]:
        from src.daw.synth_model import SynthParams, SynthParamsTable

        attributes = {
            str(param["index"]): self.synth.get_parameter(param["index"])
            for param in self.synth.get_plugin_parameters_description()
        }
        if table:
            return SynthParamsTable(**attributes)
        else:
            return SynthParams(**attributes)

    def set_patch(self, patch: list[float]) -> None:
        plugin_parameters = sorted(
            self.synth.get_plugin_parameters_description(),
            key=lambda x: x["index"],
        )

        if len(patch) != len(plugin_parameters):
            raise ValueError(
                f"Patch is not valid with {len(patch)=}. Expected"
                f" {len(plugin_parameters)} parameters."
            )

        if not all(isinstance(param, float) for param in patch):
            raise ValueError("Patch must be a list of floats")

        finalized_patch = [
            (param["index"], patch[i]) for i, param in enumerate(plugin_parameters)
        ]
        self.synth.set_patch(finalized_patch)

    def set_random_patch(self) -> list[tuple[int, float]]:
        """
        Generate random parameters for the synth.
        """
        patch = [
            (param["index"], uniform(0, 1).rvs())
            for param in self.synth.get_plugin_parameters_description()
        ]
        self.synth.set_patch(patch)
        return patch

    def render(self, midi_path: str, render_params: RenderParams) -> np.ndarray:
        """
        load the midi file and and an audio file.
        """
        self.synth.load_midi(midi_path)
        self.engine.render(render_params.duration)
        return self.engine.get_audio()
