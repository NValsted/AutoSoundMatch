import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from dawdreamer import PluginProcessor, RenderEngine
from scipy.stats import uniform

from src.daw.render_model import RenderParams
from src.utils.code_generation import get_code_gen_header, sanitize_attribute

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
            f.write("class SynthParams(SQLModel):\n\t")
            f.write("id: Optional[str] = Field(primary_key=True, default=None)\n\t")
            f.write("\n\t".join(fields) + "\n\n")
            f.write("\tclass Config:\n")
            f.write("\t\tvalidate_all = True\n\n")
            f.write('\t_auto_uuid = hash_field_to_uuid("id")\n\n\n')
            f.write("class SynthParamsTable(SynthParams, table=True):\n\t")
            f.write('__tablename__ = "SynthParams"\n\t')
            f.write(
                "__table_args__ = (UniqueConstraint(\n\t\t"
                + ",\n\t\t".join(str_field_names)
                + "\n\t),)\n"
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
