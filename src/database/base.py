from dataclasses import dataclass
from typing import Optional, TypeVar
from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.sql.schema import Table
from sqlmodel import create_engine, SQLModel, Session

ModelType = TypeVar("ModelType", bound=SQLModel)


@dataclass
class Database:
    engine: Engine

    @contextmanager
    def session():
        with Session(create_engine()) as session:
            yield session

    def create_tables(self, tables: Optional[list[Table]] = None) -> None:
        SQLModel.metadata.create_all(self.engine, tables=tables)

    def drop_tables(self, tables: Optional[list[Table]] = None) -> None:
        SQLModel.metadata.drop_all(self.engine, tables=tables)

    def add(self, instances: list[ModelType]) -> None:
        with self.session() as session:
            session.add_all(instances)
            session.commit()
