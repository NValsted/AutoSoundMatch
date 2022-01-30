from contextlib import contextmanager


@contextmanager
def temporary_attrs(cls: object, **kwargs) -> object:
    old_attrs = dict()
    try:
        for k, v in kwargs.items():
            if k in cls.__dataclass_fields__:
                old_attrs[k] = getattr(cls, k)
                setattr(cls, k, v)
        yield cls
    finally:
        for k, v in old_attrs.items():
            setattr(cls, k, v)
