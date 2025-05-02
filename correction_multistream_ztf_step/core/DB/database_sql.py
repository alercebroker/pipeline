from contextlib import contextmanager
from typing import Callable, ContextManager
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models_new import (
    Base,
    Detection,
    ZtfForcedPhotometry,
    ForcedPhotometry,
    NonDetection,
    Object,
    ZtfDetection,
)

logger = logging.getLogger(__name__)


def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


class PSQLConnection:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)
        if schema:
            self._engine = engine or create_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
            )
        else:
            self._engine = engine or create_engine(db_url, echo=False)

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


def default_parser(data, **kwargs):
    return data


def _get_sql_detections(oids: list, db_sql: PSQLConnection, parser: Callable = default_parser):
    oids = [int(oid) for oid in oids]
    if db_sql is None:
        return []
    with db_sql.session() as session:

        stmt = select(Detection).where(Detection.oid.in_(oids))
        detections = session.execute(stmt).all()

        ztf_stmt = select(ZtfDetection).where(ZtfDetection.oid.in_(oids))
        ztf_detections = session.execute(ztf_stmt).all()

        return parser(ztf_detections, detections, oids=oids)


def _get_sql_non_detections(oids, db_sql, parser: Callable = default_parser):
    oids = [int(oid) for oid in oids]
    if db_sql is None:
        return []
    with db_sql.session() as session:
        stmt = select(NonDetection).where(NonDetection.oid.in_(oids))
        detections = session.execute(stmt).all()

        return parser(detections, oids=oids)


def _get_sql_forced_photometries(oids, db_sql, parser: Callable = default_parser):
    if db_sql is None:
        return []
    oids = [int(oid) for oid in oids]
    with db_sql.session() as session:

        ztf_stmt = select(ZtfForcedPhotometry).where(ZtfForcedPhotometry.oid.in_(oids))
        ztf_forced = session.execute(ztf_stmt).all()

        stmt = select(ForcedPhotometry).where(ForcedPhotometry.oid.in_(oids))
        forced = session.execute(stmt).all()

        return parser(ztf_forced, forced, oids=oids)
