from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models_pipeline import Probability
from contextlib import contextmanager
from typing import Callable, ContextManager
from sqlalchemy import create_engine
from sqlalchemy import event
from psycopg2.extensions import quote_ident
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import logging

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
    def __init__(self, config: dict, poolclass: str | None = None) -> None:
        url = self.__format_db_url(config)
        args = self.__format_connection_args(config)

        if poolclass == "NullPool":
            poolclass = NullPool
        else:
            poolclass = None

        self._engine = create_engine(url, connect_args=args, echo=False, poolclass=poolclass)
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
    classifier_id: int,
    classifier_version: str,
    class_taxonomy: dict[str, int],
    predictions: list[dict],
) -> None:
    if len(predictions) == 0:
        return

    with psql_connection.session() as session:
        data = _format_data(classifier_id, classifier_version, class_taxonomy, predictions)

        stmt = insert(Probability)
        # the classification belongs to the first detection: only an older alert wins
        insert_stmt = stmt.on_conflict_do_update(
            constraint="pk_probability_oid_classifierid_classid",
            set_={
                "probability": stmt.excluded.probability,
                "ranking": stmt.excluded.ranking,
                "lastmjd": stmt.excluded.lastmjd,
                "classifier_version": stmt.excluded.classifier_version,
            },
            where=Probability.lastmjd > stmt.excluded.lastmjd,
        )

        session.execute(insert_stmt, data)
        session.commit()


def _format_data(
    classifier_id: int, classifier_version: str, class_taxonomy: dict[str, int], predictions: list[dict]
) -> list[dict]:
    formated_probabilities = []
    for prediction in predictions:
        probabilities = prediction["probabilities"]
        dia_object_id = prediction["diaObjectId"] #aqui tambien vienen los SSobjectid, los guardo en el step
        ss_object_id = prediction["ssObjectId"]
        alert_mjd = prediction["midpointMjdTai"]
        sid = 1 if (dia_object_id is not None and dia_object_id != 0) else 2

        # sort probabilities by value in descending order
        probabilities = sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )
        for i, (class_name, probability) in enumerate(probabilities):
            formated_probabilities.append(
                {
                    "oid": dia_object_id if dia_object_id is not None and dia_object_id != 0 else ss_object_id,
                    "sid": sid,
                    "classifier_id": classifier_id,
                    "classifier_version": classifier_version_str_to_small_integer(
                        classifier_version
                    ),
                    "class_id": class_name_to_id(class_name, class_taxonomy),
                    "probability": probability,
                    "ranking": i + 1,
                    "lastmjd": alert_mjd,  # taking midpointMjdTai from current diasource as lastmjd
                }
            )

    return formated_probabilities


# TODO: The following function is a placeholder and should be replaced with the actual implementation
def classifier_version_str_to_small_integer(version: str) -> int:
    """
    Convert a version string to a small integer.
    Example: "1.2.3" -> 123
    """
    parts = version.split(".")
    return int("".join(parts)) if len(parts) == 3 else 0


# TODO: The following function is a placeholder and should be replaced with the actual implementation
# Hay que agregar las multisuvery credentials a la config env

def class_name_to_id(class_name: str, class_taxonomy: dict[str, int]) -> int:
    """
    Convert a class name to an integer ID.
    This is a placeholder function and should be replaced with actual logic.
    """

    class_dict = class_taxonomy
    return class_dict.get(class_name, -1)  # Return -1 if class_name not found


# TODO: The following function is a placeholder and should be replaced with the actual implementation
def class_id_to_name(class_id: int, class_taxonomy: dict[str, int]) -> str:
    """
    Convert a class ID to a class name.
    This is a placeholder function and should be replaced with actual logic.
    """
    class_dict = {idx: name for name, idx in class_taxonomy.items()}
    return class_dict.get(class_id, "unknown")


def get_taxonomy_by_classifier_id(classifier_id: int, psql_connection: PSQLConnection) -> dict[str, int]:
    """Fetch taxonomy from DB for a given classifier, return {class_name: class_id}.

    Expects a table with columns: class_id, class_name, "order", classifier_id, created_date
    available under the configured schema.
    """
    mapping: dict[str, int] = {}
    try:
        with psql_connection.session() as session:
            query = text(
                """
                SELECT class_id, class_name
                FROM taxonomy
                WHERE classifier_id = :classifier_id
                ORDER BY "order" ASC
                """
            )
            result = session.execute(query, {"classifier_id": classifier_id})

            rows = result.mappings().all()
            if rows:
                mapping = {row["class_name"]: int(row["class_id"]) for row in rows}
            else:
                logging.warning(
                    f"No taxonomy rows found for classifier_id={classifier_id}."
                )
    except Exception as e:
        logging.error(f"Error fetching taxonomy for classifier_id={classifier_id}: {e}")
    
    return mapping

