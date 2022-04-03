import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union
from warnings import warn

import torch
from pydantic import BaseModel

from src.config.registry_sections import (
    DatabaseSection,
    DatasetSection,
    FlowSynthSection,
    PathSection,
    RegistrySectionsMap,
    SynthSection,
    TrainMetadataSection,
)

_REGISTRY_CONFIG_FILE = "src/config/.registry"
_BLOB_REFERENCE_FILE = "src/config/.blob_ref"


class Registry(BaseModel):
    """
    Wrapper used for accessing and manipulating the registry file.
    """

    PATH: Optional[PathSection] = None
    DATABASE: Optional[DatabaseSection] = None
    SYNTH: Optional[SynthSection] = None
    DATASET: Optional[DatasetSection] = None
    FLOWSYNTH: Optional[FlowSynthSection] = None
    TRAINMETA: Optional[TrainMetadataSection] = None

    def is_modified(self) -> bool:
        return bool(self._state_changes)

    def commit(self) -> None:
        with open(_REGISTRY_CONFIG_FILE, "wb") as f:
            pickle.dump(self, f)

    def __getattribute__(self, __name: str) -> Any:
        attr = super().__getattribute__(__name)
        if attr is None:
            warn(f"Section {__name} is not initialized")
        return attr

    def add_blob(self, blob_ref: Union[str, Path], validate: bool = True) -> None:
        if validate:
            assert os.path.exists(blob_ref), f"BLOB {blob_ref} does not exist"
        with open(_BLOB_REFERENCE_FILE, "a") as f:
            f.write(str(blob_ref) + "\n")  # TODO : maybe convert to absolute path
        # TODO : look into ways to optimize this e.g. don't add same file twice

    def clear_blobs(self) -> None:
        with open(_BLOB_REFERENCE_FILE, "r") as f:
            for line in f.readlines():
                blob_ref = line.strip()
                if os.path.exists(blob_ref):
                    try:
                        os.remove(blob_ref)
                    except PermissionError as e:
                        warn(f"Attempted to remove {blob_ref}, but got exception {e}")
                else:
                    warn(f"Attempted to remove {blob_ref}, but it does not exist")
        with open(_BLOB_REFERENCE_FILE, "w") as f:
            f.write("")

    @staticmethod
    def classify_section(section: BaseModel) -> str:
        for name, registry_section in RegistrySectionsMap.items():
            if isinstance(section, registry_section):
                return name
        raise ValueError(f"Invalid section {section}")


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
