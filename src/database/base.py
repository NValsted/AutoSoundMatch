from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import Optional, TypeVar

from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.schema import Table
from sqlmodel import Session, SQLModel

ModelType = TypeVar("ModelType", bound=SQLModel)


@dataclass
class Database:
    engine: Engine

    @contextmanager
    def session(self):
        with Session(bind=self.engine, future=True) as session:
            yield session

    def create_tables(self, tables: Optional[list[Table]] = None) -> None:
        SQLModel.metadata.create_all(self.engine, tables=tables)

    def drop_tables(self, tables: Optional[list[Table]] = None) -> None:
        SQLModel.metadata.drop_all(self.engine, tables=tables)

    def add(self, instances: list[ModelType]) -> None:
        with self.session() as session:
            session.add_all(instances)
            session.commit()

    def safe_add(self, instances: list[ModelType], timeout: int = 30) -> None:
        """
        Retries add operation with exponential backoff
        """

        start_transaction = datetime.utcnow()
        success = False
        delay = 1

        while not success:
            try:
                self.add(instances=instances)
                success = True
            except OperationalError as err:
                if (datetime.utcnow() - start_transaction) > timedelta(seconds=timeout):
                    raise err
                sleep(delay)
                delay *= 2
