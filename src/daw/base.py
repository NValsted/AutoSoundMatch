import os
from dataclasses import dataclass

from dawdreamer import RenderEngine
from dawdreamer.dawdreamer import PluginProcessor

from src.utils.code_generation import sanitize_attribute, get_code_gen_header
from src.daw.synth_model import SynthParams  # This might not work properly when switching synths


@dataclass
class SynthHost:
    engine: RenderEngine
    synth: PluginProcessor
    _synth_model_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/synth_model.py"

    def create_parameter_table(self) -> str:
        sorted_parameters = sorted(
            self.synth.get_plugin_parameters_description(), key=lambda x: x["index"]
        )
        fields = [
            f'{sanitize_attribute(param["name"])}: float = Field(alias="{param["index"]}")'
            for param in sorted_parameters
        ]

        assert len(fields) == len(set(fields))

        str_field_names = map(lambda x: '"{}"'.format(x.split(":")[0]), fields)

        with open(self._synth_model_path, "w") as f:
            f.write(get_code_gen_header())
            f.write("from typing import Optional\n\n")
            f.write("from sqlmodel import SQLModel, Field\n")
            f.write("from sqlalchemy import UniqueConstraint\n\n\n")
            f.write("class SynthParams(SQLModel):\n\t")
            f.write("id: Optional[int] = Field(primary_key=True, default=None)\n\t")
            f.write("\n\t".join(fields) + "\n\n")
            f.write("class SynthParamsTable(SynthParams, table=True):\n\t")
            f.write("__tablename__ = \"SynthParams\"\n\t")
            f.write(
                "__table_args__ = "
                "(UniqueConstraint(\n\t\t"+ ",\n\t\t".join(str_field_names) + "\n\t),)\n"
            )

        return self._synth_model_path

    def set_random_patch():
        """
        Generate random parameters for the synth.
        """
        pass
