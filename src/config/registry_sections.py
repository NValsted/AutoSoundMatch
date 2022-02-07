from typing import Optional
from enum import Enum, unique

from pydantic import BaseModel


class DatabaseSection(BaseModel):
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


@unique
class RegistrySectionsEnum(str, Enum):
    DATABASE = "DATABASE"


RegistrySectionsMap = dict(
    DATABASE=DatabaseSection,
)
