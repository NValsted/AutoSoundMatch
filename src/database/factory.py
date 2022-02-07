from dataclasses import dataclass

from sqlmodel import create_engine

from src.database.base import Database
from src.utils.temporary_context import temporary_attrs

from src.config.base import Registry
from src.config.registry_sections import DatabaseSection


@dataclass
class DBFactory:
    engine_url: str = "sqlite://"

    def __call__(self, *args, **kwargs) -> Database:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            engine = create_engine(url=tmp.engine_url, **kwargs)
            return Database(engine=engine, **kwargs)

    def register(self):
        section = DatabaseSection(
            url=self.engine_url,
        )
        registry = Registry()
        registry.DATABASE = section
        registry.commit()
