from typing import Optional

from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class RenderParams(SQLModel):
    id: Optional[int] = Field(primary_key=True, default=None)
    sample_rate: int = Field(index=False)
    buffer_size: int = Field(index=False)
    bpm: int = Field(index=False)
    duration: int = Field(index=False)


class RenderParamsTable(RenderParams, table=True):
    __tablename__ = "RenderParams"
    __table_args__ = (UniqueConstraint("sample_rate", "buffer_size", "bpm", "duration"),)
