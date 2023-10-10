from contextlib import contextmanager
from typing import Callable, ContextManager, Dict

from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


class DatabaseConnection:
    def __init__(self, config: dict):
        self.config = config
        _database = self.config.pop("database")
        self.client = MongoClient(**self.config)
        self.database = self.client[_database]


class PSQLConnection:
    def __init__(self, config: Dict) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url)
        self._session_factory = sessionmaker(self._engine)

    def __format_db_url(self, config: Dict):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

    @contextmanager
    def session(self) -> Callable[[], ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
