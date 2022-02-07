import pickle
from warnings import warn
from typing import Any, Optional

from pydantic import BaseModel

from src.utils.meta import Singleton
from src.config.registry_sections import DatabaseSection

REGISTRY_CONFIG_FILE = "src/config/.registry"


class Registry(BaseModel, Singleton):
    """
    Wrapper used for accessing and manipulating the registry file.
    """
    DATABASE: Optional[DatabaseSection] = None

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
    if len(content) == 0:
        Registry()
    else:
        try:
            pickle.loads(content)
        except pickle.UnpicklingError:
            warn(f"Registry file is corrupted, resetting")
            Registry()
