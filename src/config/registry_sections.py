from typing import Optional
from enum import Enum, unique

from pydantic import BaseModel


class DatabaseSection(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


class SynthSection(BaseModel):
    synth_path: str
    sample_rate: int = 44100
    buffer_size: int = 128
    bpm: int = 128
    duration: float = 1.0


@unique
class RegistrySectionsEnum(str, Enum):
    DATABASE = "DATABASE"
    SYNTH = "SYNTH"


RegistrySectionsMap = dict(
    DATABASE=DatabaseSection,
    SYNTH=SynthSection,
)
