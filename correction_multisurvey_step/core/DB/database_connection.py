from contextlib import contextmanager
from typing import Callable, ContextManager
import logging
from sqlalchemy.pool import NullPool


from sqlalchemy import create_engine, select
from sqlalchemy import event
from psycopg2.extensions import quote_ident
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)


def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"

def _set_search_path_on_connect(engine, schema):
    """Apply `schema` with a SET on every new connection.

    The `-csearch_path` startup option is dropped by PgBouncer
    (ignore_startup_parameters=options), which left sessions on the DB user's
    default search_path. A SET after connecting is passed through, and holds for
    the whole connection in PgBouncer's session mode.
    """
    schemas = [s.strip() for s in schema.split(",") if s.strip()]

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute(
            "SET search_path TO " + ", ".join(quote_ident(s, cursor) for s in schemas)
        )
        cursor.close()
        dbapi_connection.commit()  # a later rollback would otherwise undo the SET


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

        if schema and engine is None:
            _set_search_path_on_connect(self._engine, schema)

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

