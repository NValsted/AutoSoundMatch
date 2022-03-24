import os
import pickle
from typing import Any, Optional
from warnings import warn

import torch
from pydantic import BaseModel

from src.config.registry_sections import (
    DatabaseSection,
    DatasetSection,
    FlowSynthSection,
    RegistrySectionsEnum,
    RegistrySectionsMap,
    SynthSection,
    TrainMetadataSection,
)

_REGISTRY_CONFIG_FILE = "src/config/.registry"
_BLOB_REFERENCE_FILE = "src/config/.blob_ref"


class SectionDefaultNone:
    _section: RegistrySectionsEnum

    def __init__(self, section: RegistrySectionsEnum):
        self._section = section

    def __getattribute__(self, __name: str) -> Any:
        if __name == "_section":
            return super().__getattribute__(__name)
        warn(
            f"{self._section.value} is not initialized. Attempting to retrieve default"
            f" value for '{__name}'"
        )

        fields = RegistrySectionsMap[self._section.value].__fields__
        if __name not in fields:
            raise AttributeError(
                f"{__name} is not a valid attribute for {self._section.value}"
            )
        if fields[__name].required:
            raise AttributeError(f"{self._section.value}.{__name} has no default value")

        return fields[__name].default

    def __nonzero__(self) -> bool:
        return False

    def __bool__(self) -> bool:
        return bool(None)

    def __lt__(self, __o: object) -> bool:
        return False

    def __eq__(self, __o: object) -> bool:
        return None == __o  # NOQA: E711

    def __hash__(self) -> int:
        return hash(None)


class Registry(BaseModel):
    """
    Wrapper used for accessing and manipulating the registry file.
    """

    DATABASE: Optional[DatabaseSection] = None
    SYNTH: Optional[SynthSection] = None
    DATASET: Optional[DatasetSection] = None
    FLOWSYNTH: Optional[FlowSynthSection] = None
    TRAINMETA: Optional[TrainMetadataSection] = None

    def commit(self) -> None:
        with open(_REGISTRY_CONFIG_FILE, "wb") as f:
            pickle.dump(self, f)

    def __getattribute__(self, __name: str) -> Any:
        attr = super().__getattribute__(__name)

        if attr is None:
            warn(f"Section {__name} is not initialized")
            if __name in RegistrySectionsEnum.__members__:
                return SectionDefaultNone(RegistrySectionsEnum(__name))

        return attr

    def add_blob(self, blob_ref: str, validate: bool = True) -> None:
        if validate:
            assert os.path.exists(blob_ref), f"BLOB {blob_ref} does not exist"
        with open(_BLOB_REFERENCE_FILE, "a") as f:
            f.write(blob_ref + "\n")  # TODO : maybe convert to absolute path
        # TODO : look into ways to optimize this e.g. don't add same file twice

    def clear_blobs(self) -> None:
        with open(_BLOB_REFERENCE_FILE, "r") as f:
            for line in f.readlines():
                blob_ref = line.strip()
                if os.path.exists(blob_ref):
                    os.remove(blob_ref)
                else:
                    warn(f"Attempted to remove {blob_ref}, but it does not exist")
        with open(_BLOB_REFERENCE_FILE, "w") as f:
            f.write("")


for file in [_REGISTRY_CONFIG_FILE, _BLOB_REFERENCE_FILE]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("")


with open(_REGISTRY_CONFIG_FILE, "rb") as f:
    content = f.read()
    try:
        REGISTRY = pickle.loads(content) if content else Registry()
    except pickle.UnpicklingError:
        warn("Registry file is corrupted, resetting")
        REGISTRY = Registry()

PYTORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
