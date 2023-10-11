from contextlib import contextmanager

from pymongo import MongoClient
from typing import Iterator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager


class MongoConnection:
    def __init__(self, config: dict):
        self.config = config
        _database = self.config.pop("database")
        self.client = MongoClient(**self.config)
        self.database = self.client[_database]


class PsqlConnection:
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=True)
        self._session_factory = sessionmaker(
            self._engine,
        )

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

    @contextmanager
    def session(self) -> Iterator[Session]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise Exception(e)
        finally:
            session.close()
