from contextlib import contextmanager
from typing import Callable, ContextManager
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


class PSQLConnection:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config)
        poolclass_name = db_config.get("POOLCLASS", None)
        poolclass = NullPool if poolclass_name == "NullPool" else None
        self._engine = engine or create_engine(db_url, echo=False, poolclass=poolclass)

        self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            logger.exception("Session rollback because of exception")
            logger.exception(e)
            session.rollback()
            raise Exception(e)
        finally:
            session.close()

