from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from src.utils.meta import hash_field_to_uuid


class TrainValTestEnum(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class Loss(SQLModel):
    id: Optional[str] = Field(primary_key=True, default=None)
    model_id: Optional[str] = Field(index=True, default=None)
    type: Optional[str] = Field(default=None)
    train_val_test: TrainValTestEnum
    value: float
    time: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        validate_all = True

    _auto_uuid = hash_field_to_uuid("id")


class LossTable(Loss, table=True):
    __tablename__ = "Loss"
    __table_args__ = (
        UniqueConstraint("model_id", "type", "train_val_test", "value", "time"),
    )
