from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import Detection, NonDetection


class PSQLConnection:
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=True)
        self._session_factory = sessionmaker(
            self._engine,
        )

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


def default_parser(data, **kwargs):
    return data


def _get_sql_detections(
    oids: dict, db_sql: PSQLConnection, parser: Callable = default_parser
):
    with db_sql.session() as session:
        stmt = select(Detection).where(Detection.oid.in_(oids))
        detections = session.execute(stmt).all()
        return parser(detections, oids=oids)


def _get_sql_non_detections(oids, db_sql, parser: Callable = default_parser):
    with db_sql.session() as session:
        stmt = select(NonDetection).where(NonDetection.oid.in_(oids))
        detections = session.execute(stmt).all()
        return parser(detections, oids=oids)


def _get_sql_forced_photometries(oids, db_sql, parser: Callable = default_parser):
    return
