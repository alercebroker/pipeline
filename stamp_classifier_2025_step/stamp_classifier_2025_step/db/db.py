from alerce_classifiers.base.dto import OutputDTO
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models_pipeline import Probability
from contextlib import contextmanager
from typing import Callable, ContextManager
from sqlalchemy import create_engine
from sqlalchemy import event
from psycopg2.extensions import quote_ident
from sqlalchemy.orm import sessionmaker, Session

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
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        args = self.__format_connection_args(config)
        
        self._engine = create_engine(url, connect_args=args, echo=False)
        _set_search_path_on_connect(self._engine, config["SCHEMA"])
        self._session_factory = sessionmaker(
            self._engine,
        )

    def __format_db_url(self, config):
        return f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
    
    def __format_connection_args(self, config):
        return {"options": "-csearch_path={}".format(config["SCHEMA"])}
    
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

def store_probability(
        psql_connection: PSQLConnection,
        classifier_name: str,
        classifier_version: str,
        output_dto: OutputDTO
    ):

    # early exit
    if output_dto.probabilities.shape[0] == 0:
        return

    with psql_connection.session() as session:
            
        data = _format_data(classifier_name, classifier_version, output_dto)

        insert_stmt = insert(Probability)
        insert_stmt = insert_stmt.on_conflict_do_nothing()

        session.execute(insert_stmt, data)
        session.commit()

def _format_data(
        classifier_name: str,
        classifier_version: str,
        output_dto: OutputDTO
    ) -> list[dict]: 

    probabilitites = output_dto.probabilities.reset_index()
    probabilitites_melt = probabilitites.melt(
        id_vars=["oid"], var_name="class_name", value_name="probability"
    )
    probabilitites_melt["ranking"] = probabilitites_melt.groupby("oid")[
        "probability"
    ].rank(ascending=False, method="dense")

    probabilitites_melt["classifier_name"] = classifier_name
    probabilitites_melt["classifier_version"] = classifier_version
    
    formated_probabilities = probabilitites_melt.to_dict(orient="records")

    return formated_probabilities

    
