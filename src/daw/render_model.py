from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from src.utils.meta import hash_field_to_uuid


class RenderParams(SQLModel):
    id: Optional[str] = Field(primary_key=True, default=None)
    sample_rate: int = Field(index=False, default=22050)
    buffer_size: int = Field(index=False, default=128)
    bpm: int = Field(index=False, default=128)
    duration: float = Field(index=False, default=4.0)

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")


class RenderParamsTable(RenderParams, table=True):
    __tablename__ = "RenderParams"
    __table_args__ = (
        UniqueConstraint("sample_rate", "buffer_size", "bpm", "duration"),
    )
