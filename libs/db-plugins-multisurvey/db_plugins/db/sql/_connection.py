import logging
from contextlib import (
    asynccontextmanager,
    contextmanager,
)
from typing import Callable, ContextManager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def get_db_url(config: dict, is_async: bool = False):
    protocol = "postgresql"
    if is_async:
        protocol += "+psycopg"

    return f"{protocol}://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


logger = logging.getLogger(__name__)


class PsqlDatabase:
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

        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def create_db(self):
        Base.metadata.create_all(self._engine)

    def drop_db(self):
        Base.metadata.drop_all(self._engine)

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


class AsyncPsqlDatabase:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config, is_async=True)
        schema = db_config.get("SCHEMA", None)
        if schema:
            self._engine = engine or create_async_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
            )
        else:
            self._engine = engine or create_async_engine(db_url, echo=False)

        self._session_factory = async_sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    async def create_db(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_db(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        session = self._session_factory()
        try:
            yield session
        except Exception as e:
            logger.exception("Session rollback because of exception")
            logger.exception(e)
            await session.rollback()
            raise
        finally:
            await session.close()

