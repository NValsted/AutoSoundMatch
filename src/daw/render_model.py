from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class RenderParams(SQLModel):
    id: Optional[int] = Field(primary_key=True, default=None)
    sample_rate: int = Field(index=False, default=44100)
    buffer_size: int = Field(index=False, default=128)
    bpm: int = Field(index=False, default=128)
    duration: float = Field(index=False, default=2.0)


class RenderParamsTable(RenderParams, table=True):
    __tablename__ = "RenderParams"
    __table_args__ = (UniqueConstraint("sample_rate", "buffer_size", "bpm", "duration"),)
