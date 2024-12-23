from contextlib import contextmanager
from typing import Callable, ContextManager, List, Optional

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import Reference


class PSQLConnection:
    def __init__(self, config: dict, echo=False) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=echo)
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


def parse_sql_reference(reference_models: list, keys) -> pd.DataFrame:
    reference_info = []
    for ref in reference_models:
        ref = ref[0].__dict__
        reference_info.append([ref[k] for k in keys])
    return pd.DataFrame(reference_info, columns=keys)


def get_sql_references(
    oids: List[str], db_sql: PSQLConnection, keys: List[str]
) -> Optional[pd.DataFrame]:
    if db_sql is None:
        return None
    with db_sql.session() as session:
        stmt = select(Reference).where(Reference.oid.in_(oids))
        reference = session.execute(stmt).all()
        df = parse_sql_reference(reference, keys)
        return df
