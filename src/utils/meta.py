from threading import Lock
from uuid import NAMESPACE_DNS, UUID, uuid5

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
