import pickle
from warnings import warn
from typing import Any, Optional

from pydantic import BaseModel

from src.config.registry_sections import DatabaseSection, SynthSection, DatasetSection, FlowSynthSection, TrainMetadataSection

REGISTRY_CONFIG_FILE = "src/config/.registry"


class Registry(BaseModel):
    """
    Wrapper used for accessing and manipulating the registry file.
    """
    DATABASE: Optional[DatabaseSection] = None
    SYNTH: Optional[SynthSection] = None
    DATASET: Optional[DatasetSection] = None
    FLOWSYNTH: Optional[FlowSynthSection] = None
    TRAINMETA: Optional[TrainMetadataSection] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_modified(self) -> bool:
        return bool(self._state_changes)

    def commit(self) -> None:
        with open(REGISTRY_CONFIG_FILE, "wb") as f:
            pickle.dump(self, f)

    def __getattribute__(self, __name: str) -> Any:
        attr = super().__getattribute__(__name)
        if attr is None:
            warn(f"Section {__name} is not initialized")
        return attr


with open(REGISTRY_CONFIG_FILE, "rb") as f:
    content = f.read()    
    try:
        REGISTRY = pickle.loads(content) if content else Registry()
    except pickle.UnpicklingError:
        warn(f"Registry file is corrupted, resetting")
        REGISTRY = Registry()
