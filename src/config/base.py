from typing import Any, Union
from warnings import warn
from dataclasses import dataclass
from configparser import ConfigParser, SectionProxy

from src.utils.meta import Singleton

REGISTRY_CONFIG_FILE = "src/config/.registry.cfg"
SECTION_SEPARATOR = ";"


@dataclass
class StateChange:
    previous: Any = None
    current: Any = None


class Registry(Singleton, ConfigParser):
    """
    Wrapper used for accessing and manipulating the registry file.
    """
    _state_changes: dict[StateChange]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read(REGISTRY_CONFIG_FILE)
        self._state_changes = dict()

    def is_modified(self) -> bool:
        return len(self._state_changes) > 0

    def commit(self) -> None:
        print(self._state_changes)
        with open(REGISTRY_CONFIG_FILE, "w") as f:
            self.write(f)
        self._state_changes = dict()

    def refresh(self) -> None:
        if self.is_modified():
            warn("Refreshing the registry without committing changes first")
        self.read(REGISTRY_CONFIG_FILE)
        self._state_changes = dict()

    def __getitem__(self, section: str) -> SectionProxy:
        if SECTION_SEPARATOR in section:
            section, subkey = section.split(SECTION_SEPARATOR)
            return super().__getitem__(section)[subkey]
        return super().__getitem__(section)

    def __setitem__(self, section: str, options: dict[str, Any]) -> None:
        subkey, value = options.popitem()
        key = f"{section}{SECTION_SEPARATOR}{subkey}"
        
        state_change = StateChange(previous=None, current=value)
        
        try:
            state_change.previous = self[key]
            if key in self._state_changes:
                state_change.previous = self._state_changes[key].previous
        except KeyError:
            pass
        
        if state_change.previous != state_change.current:
            self._state_changes[key] = state_change
        elif key in self._state_changes:
            del self._state_changes[key]

        return super().__setitem__(section, options)
