from functools import partial
from itertools import chain
from pathlib import Path
from typing import Union

from pydantic import Field


def get_project_root() -> Path:
    cwd = Path.cwd()

    for directory in chain((cwd,), cwd.parents):
        if directory.name == "AutoSoundMatch":
            return directory

    matcher = Path(".").glob("**/AutoSoundMatch")
    for match in matcher:
        project_root = Path(".") / match
        return project_root.resolve()

    raise FileNotFoundError("Could not determine project root directory")


def path_factory(path: Union[str, Path], mkdir: bool = False) -> Path:
    resolved_path = Path(path).resolve()
    if mkdir:
        resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def path_field_factory(path: Union[str, Path]) -> Field:
    default_factory = partial(path_factory, path=path.resolve(), mkdir=True)
    return Field(default_factory=default_factory)
