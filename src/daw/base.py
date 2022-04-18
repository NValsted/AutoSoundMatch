import json
from dataclasses import dataclass
from pathlib import Path
from re import IGNORECASE
from re import compile as compile_regex
from typing import TYPE_CHECKING, Union

import numpy as np
from dawdreamer import PluginProcessor, RenderEngine
from scipy.stats import uniform

from src.config.base import REGISTRY
from src.utils.code_generation import INDENT, get_code_gen_header, sanitize_attribute

if TYPE_CHECKING:
    from src.daw.synth_model import SynthParams, SynthParamsTable

MIDI_CC_PATTERN = compile_regex(r"^MIDI\s*CC\s*\d*\|\d*$", flags=IGNORECASE)


@dataclass
class SynthHost:
    engine: RenderEngine
    synth: PluginProcessor
    locked_parameters: dict[int, float]
    _synth_model_path: Path = (
        REGISTRY.PATH.project_root / "src" / "daw" / "synth_model.py"
    )

    def _get_sorted_parameters(
        self, include_midi_cc: bool = False, include_locked: bool = False
    ) -> list[dict]:
        sorted_parameters = sorted(
            [
                param
                for param in self.synth.get_plugin_parameters_description()
                if include_midi_cc or not MIDI_CC_PATTERN.match(param["name"])
            ],
            key=lambda x: x["index"],
        )
        finalized_parameters = [
            param
            for idx, param in enumerate(sorted_parameters)
            if idx not in self.locked_parameters or include_locked
        ]
        return finalized_parameters

    def create_parameter_table(
        self, include_midi_cc: bool = False, include_locked: bool = False
    ) -> Path:
        sorted_parameters = self._get_sorted_parameters(
            include_midi_cc=include_midi_cc, include_locked=include_locked
        )
        fields = [
            f'{sanitize_attribute(param["name"])}_{param["index"]}: float ='
            f' Field(alias="{param["index"]}")'
            for param in sorted_parameters
        ]

        assert len(fields) == len(set(fields))

        with self._synth_model_path.open("w") as f:
            f.write(get_code_gen_header())
            f.write("from typing import Optional\n\n")
            f.write("from sqlmodel import SQLModel, Field\n\n")
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
            f.write('__tablename__ = "SynthParams"\n')

        return self._synth_model_path

    def get_patch_as_model(
        self,
        table: bool = False,
        include_midi_cc: bool = False,
        include_locked: bool = False,
    ) -> Union["SynthParams", "SynthParamsTable"]:
        from src.daw.synth_model import SynthParams, SynthParamsTable

        attributes = {
            str(param["index"]): self.synth.get_parameter(param["index"])
            for param in self._get_sorted_parameters(
                include_midi_cc=include_midi_cc, include_locked=include_locked
            )
        }

        if table:
            return SynthParamsTable(**attributes)
        else:
            return SynthParams(**attributes)

    def _enforce_locked_parameters(
        self,
        patch: list[Union[float, tuple[int, float]]],
        include_midi_cc: bool = False,
        include_locked: bool = False,
    ) -> list[Union[float, tuple[int, float]]]:
        parameters = self._get_sorted_parameters(
            include_midi_cc=include_midi_cc, include_locked=include_locked
        )
        if len(parameters) != len(patch):
            raise ValueError(
                f"Patch is not valid with {len(patch)=}. Expected"
                f" {len(parameters)} parameters."
            )

        if include_locked:
            for param_idx, value in self.locked_parameters.items():
                if isinstance(patch[param_idx], tuple):
                    patch[param_idx] = (patch[param_idx][0], value)
                else:
                    patch[param_idx] = value
        else:
            for param in patch:
                if isinstance(param, tuple):
                    if param[0] in self.locked_parameters:
                        raise ValueError(
                            f"Locked parameter {param} present in patch, but"
                            f" {include_locked=}"
                        )
        return patch

    def _load_json_preset(self, patch_path: Path) -> list[float]:
        with patch_path.open("r") as f:
            patch_dict = json.load(f)

        params = self._get_sorted_parameters(include_locked=True)
        params_midi_cc = self._get_sorted_parameters(
            include_midi_cc=True, include_locked=True
        )
        if len(patch_dict) == len(params):
            patch = [patch_dict[param["name"]] for param in params]
        elif len(patch_dict) == len(params_midi_cc):
            patch = [patch_dict[param["name"]] for param in params_midi_cc]
        else:
            raise ValueError(
                f"Patch file {patch_path} has an invalid number of parameters"
            )
        return patch

    def _load_fxp_preset(self, patch_path: Path) -> list[float]:
        self.synth.load_preset(str(patch_path))
        patch = [
            self.synth.get_parameter(param["index"])
            for param in self._get_sorted_parameters(
                include_locked=True,
            )
        ]
        return patch

    def _load_vst3_preset(self, patch_path: Path) -> list[float]:
        self.synth.load_vst3_preset(str(patch_path))
        patch = [
            self.synth.get_parameter(param["index"])
            for param in self._get_sorted_parameters(
                include_locked=True,
            )
        ]
        return patch

    def load_preset(
        self,
        patch_path: Path,
        include_midi_cc: bool = False,
        include_locked: bool = False,
    ) -> list[float]:
        if patch_path.suffix == ".json":
            patch = self._load_json_preset(patch_path)
        elif patch_path.suffix == ".fxp":
            patch = self._load_fxp_preset(patch_path)
        elif patch_path.suffix == ".vstpreset":
            patch = self._load_vst3_preset(patch_path)
        else:
            raise ValueError(f"Unknown preset file type: {patch_path.suffix}")

        plugin_parameters = self._get_sorted_parameters(
            include_midi_cc=include_midi_cc, include_locked=include_locked
        )

        patch = self._enforce_locked_parameters(
            [patch[i] for i, _ in enumerate(plugin_parameters)],
            include_midi_cc=include_midi_cc,
            include_locked=include_locked,
        )
        return patch

    def set_patch(
        self,
        patch: list[float],
        include_midi_cc: bool = False,
        include_locked: bool = False,
    ) -> None:
        plugin_parameters = self._get_sorted_parameters(
            include_midi_cc=include_midi_cc, include_locked=include_locked
        )

        if len(patch) != len(plugin_parameters):
            raise ValueError(
                f"Patch is not valid with {len(patch)=}. Expected"
                f" {len(plugin_parameters)} parameters."
            )

        if not all(isinstance(param, float) for param in patch):
            raise ValueError("Patch must be a list of floats")

        finalized_patch = self._enforce_locked_parameters(
            [(param["index"], patch[i]) for i, param in enumerate(plugin_parameters)],
            include_midi_cc=include_midi_cc,
            include_locked=include_locked,
        )
        self.synth.set_patch(finalized_patch)

    def random_patch(self, apply: bool = False) -> list[tuple[int, float]]:
        """
        Generate random parameters for the synth.
        """
        patch = self._enforce_locked_parameters(
            [
                (param["index"], uniform(0, 1).rvs())
                for param in self._get_sorted_parameters()
            ]
        )
        if apply:
            self.synth.set_patch(patch)
        return patch

    def render(self, midi_path: Union[Path, str]) -> np.ndarray:
        """
        load the midi file and and an audio file.
        """
        if not Path(midi_path).is_file():
            raise ValueError(f"{midi_path} does not exist or is not a file")
        self.synth.load_midi(str(midi_path))
        self.engine.render(REGISTRY.SYNTH.duration)
        return self.engine.get_audio().transpose()
