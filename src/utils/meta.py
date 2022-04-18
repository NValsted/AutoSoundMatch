import io
from pickle import Unpickler
from threading import Lock
from uuid import NAMESPACE_DNS, UUID, uuid5

import torch
from pydantic import root_validator


# References https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html  # NOQA: E501
class Singleton:
    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if Singleton._instance is None:
                Singleton._instance = object.__new__(cls)
            return Singleton._instance


class AttributeWrapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def hash_to_uuid(key) -> str:
    try:
        UUID(key)
        return str(key)
    except ValueError:
        return str(uuid5(NAMESPACE_DNS, str(key)))


def hash_field_to_uuid(field: str = "id") -> str:
    @root_validator(allow_reuse=True)
    @classmethod
    def _func(cls, values):
        if values.get(field) is None:
            fields = {k: v.default for k, v in cls.__fields__.items() if k != field}
            key = "".join(
                [
                    f"{k}{v}"
                    if k not in values or values[k] is None
                    else f"{k}{values[k]}"
                    for k, v in fields.items()
                ]
            )
        else:
            key = values[field]

        values[field] = hash_to_uuid(key)
        return values

    return _func


class CPUTorchUnpickler(Unpickler):
    """
    References https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
