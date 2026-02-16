from contextlib import contextmanager
from typing import Callable, ContextManager, List, Optional

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import ztf_reference


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
        stmt = select(ztf_reference).where(ztf_reference.oid.in_(oids))
        reference = session.execute(stmt).all()
        df = parse_sql_reference(reference, keys)
        return df


def get_feature_name_lut(db_sql: PSQLConnection, schema: str, sid: int, tid: int, logger=None) -> dict:
    """Fetch feature name lookup table from multisurvey schema for LSST survey."""
    if db_sql is None:
        if logger:
            logger.warning("No database connection available for feature name lookup")
        return {}
    
    try:
        from sqlalchemy import text
        
        with db_sql.session() as session:
            # Query the feature_name_lut table from configured schema filtering by sid and tid
            query = text(
                f"SELECT feature_id, feature_name FROM {schema}.feature_name_lut WHERE sid = :sid AND tid = :tid ORDER BY feature_id"
            )
            result = session.execute(query, {"sid": sid, "tid": tid})
            
            # Create dictionary with id as key and name as value
            feature_lut = {row[0]: row[1] for row in result.fetchall()}
            
            if logger:
                logger.info(f"Loaded {len(feature_lut)} feature names from lookup table for sid={sid}, tid={tid}")
            return feature_lut
            
    except Exception as e:
        if logger:
            logger.error(f"Error fetching feature name lookup table: {e}")
        return {}


def get_or_create_version_id(db_sql: PSQLConnection, schema: str, version_name: str, sid: int, tid: int, logger=None) -> int:
    """Get version_id from version_lut table, or create it if it doesn't exist."""
    if db_sql is None:
        if logger:
            logger.warning("No database connection available for version lookup")
        return None
        
    try:
        from sqlalchemy import text
        
        with db_sql.session() as session:
            # First, try to get existing version_id
            select_query = text(
                f"SELECT version_id FROM {schema}.feature_version_lut WHERE version_name = :version_name AND sid = :sid AND tid = :tid"
            )
            result = session.execute(select_query, {"version_name": version_name, "sid": sid, "tid": tid})
            row = result.fetchone()
            
            if row:
                version_id = row[0]
                if logger:
                    logger.info(f"Found existing version_id {version_id} for version_name '{version_name}', sid={sid}, tid={tid}")
                return version_id
            else:
                # Insert new version_name, sid, tid and get the generated version_id
                insert_query = text(
                    f"INSERT INTO {schema}.feature_version_lut (version_name, sid, tid) VALUES (:version_name, :sid, :tid) RETURNING version_id"
                )
                result = session.execute(insert_query, {"version_name": version_name, "sid": sid, "tid": tid})
                version_id = result.fetchone()[0]
                session.commit()
                
                if logger:
                    logger.info(f"Created new version_id {version_id} for version_name '{version_name}', sid={sid}, tid={tid}")
                return version_id
                
    except Exception as e:
        if logger:
            logger.error(f"Error handling version_lut table: {e}")
        return 1 # None
