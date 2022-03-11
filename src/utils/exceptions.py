from contextlib import contextmanager
from inspect import stack
from traceback import format_exc


@contextmanager
def dev_dependency_hint():
    try:
        yield None
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"{format_exc()}NOTE: {stack()[0][3]} requires dev-dependencies. Check"
            " pyproject.toml"
        )


@contextmanager
def proprietary_resource_hint():
    try:
        yield None
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{format_exc()}NOTE: {stack()[0][3]} requires proprietary resources. Check"
            " README.md"
        )
