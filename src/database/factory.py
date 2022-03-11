from dataclasses import dataclass

from sqlmodel import create_engine

from src.config.base import REGISTRY
from src.config.registry_sections import DatabaseSection
from src.database.base import Database
from src.utils.temporary_context import temporary_attrs


@dataclass
class DBFactory:
    engine_url: str = "sqlite://"

    def __call__(self, *args, **kwargs) -> Database:
        with temporary_attrs(self, *args, **kwargs) as tmp:
            engine = create_engine(url=tmp.engine_url, **kwargs)
            return Database(engine=engine, **kwargs)

    def register(self, commit: bool = False) -> None:
        REGISTRY.DATABASE = DatabaseSection(
            url=self.engine_url,
        )
        if commit:
            REGISTRY.commit()
