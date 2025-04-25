from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models_new import Base
from contextlib import contextmanager
import logging

def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


logger = logging.getLogger(__name__)


class PsqlDatabase:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config)
        self._engine = engine or create_engine(db_url, echo=False)
        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def create_db(self):
        Base.metadata.create_all(self._engine)

    def drop_db(self):
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self):
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