import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from db_plugins.db.sql._initial_data_archive import INITIAL_DATA

from .models_archive_probability import Base, Taxonomy


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
        Base.metadata.create_all(self._engine)

        with self._engine.connect() as conn:
            for mapper in Base.registry.mappers:
                table_class = mapper.class_
                if getattr(table_class, "__n_partitions__", None):
                    table_class.__create_partitions__(conn, self.schema)

        self.insert_initial_data()

    def insert_initial_data(self):
        model_tables = Base.metadata.tables

        with self._engine.connect() as conn:
            for table_name, init_data in INITIAL_DATA.items():
                if table_name not in model_tables:
                    continue
                stmt = (
                    insert(model_tables[table_name])
                    .values(init_data["data"])
                    .on_conflict_do_nothing(index_elements=init_data["index_elements"])
                )
                conn.execute(stmt)
            conn.commit()

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