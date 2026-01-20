from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
from contextlib import contextmanager
from typing import Callable, ContextManager
import logging
from sqlalchemy import text


def get_db_url(config: dict):
    return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"


logger = logging.getLogger(__name__)


class PsqlDatabase:
    def __init__(self, db_config: dict, engine=None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)
        self.schema = schema
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
        # Create all tables EXCEPT feature
        tables_to_create = [
            table for table in Base.metadata.tables.values() 
            if table.name != 'feature'
        ]
        Base.metadata.create_all(self._engine, tables=tables_to_create)
        # Create features table with partitions directly using SQL
        self._create_feature_table()
        self.create_feature_partitions()
    
    def _create_feature_table(self):
        with self._engine.connect() as conn:
            schema_prefix = f"{self.schema}." if self.schema else ""
            conn.execute(text(f"DROP TABLE IF EXISTS {schema_prefix}feature CASCADE"))
            
            conn.execute(text(f"""
                CREATE TABLE {schema_prefix}feature (
                    oid bigint NOT NULL,
                    sid smallint NOT NULL,
                    feature_id smallint NOT NULL,
                    band smallint NOT NULL,
                    version smallint NOT NULL,
                    value double precision,
                    updated_date date
                ) PARTITION BY HASH (oid)
            """))
            
            conn.execute(text(f"CREATE INDEX ON {schema_prefix}feature USING btree (oid)"))
            
            conn.commit()
    
    def create_feature_partitions(self):
        """Create the 10 partitions for feature table"""
        with self._engine.connect() as conn:
            schema_prefix = f"{self.schema}." if self.schema else ""
            
            for i in range(10):
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {schema_prefix}feature_part_{i} 
                    PARTITION OF {schema_prefix}feature 
                    FOR VALUES WITH (MODULUS 10, REMAINDER {i})
                """))
                
            conn.commit()


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