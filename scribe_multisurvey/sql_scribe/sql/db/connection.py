from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

class PSQLConnection:
    def __init__(self, db_config: dict, engine=None, poolclass: str | None = None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)

        if poolclass == "NullPool":
            poolclass = NullPool
        else:
            poolclass = None

        if schema:
            self._engine = engine or create_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
                poolclass=poolclass,
            )
        else:
            self._engine = engine or create_engine(db_url, echo=False, poolclass=poolclass)

        self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
