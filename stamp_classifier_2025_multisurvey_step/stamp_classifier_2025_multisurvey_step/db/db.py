from alerce_classifiers.base.dto import OutputDTO
from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models import Probability
from contextlib import contextmanager
from typing import Callable, ContextManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text
import logging


class PSQLConnection:
    def __init__(self, config: dict) -> None:
        url = self.__format_db_url(config)
        args = self.__format_connection_args(config)
        
        self._engine = create_engine(url, connect_args=args, echo=False)
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
    sid: int,
    classifier_id: int,
    classifier_version: int,
    class_taxonomy: dict[str, int],
    output_dto: OutputDTO,
    messages_dict: dict,
):
    if output_dto.probabilities.shape[0] == 0:
        return

    with psql_connection.session() as session:
        data = format_probability_records(sid, classifier_id, classifier_version, class_taxonomy, output_dto, messages_dict)
        insert_stmt = insert(Probability).on_conflict_do_nothing()
        session.execute(insert_stmt, data)
        session.commit()

def format_probability_records(
    sid: int,
    classifier_id: int,
    classifier_version: int,
    class_taxonomy: dict[str, int],
    output_dto: OutputDTO,
    messages_dict: dict,
) -> list[dict]:
    
    probabilities = output_dto.probabilities.reset_index()

    probabilities_melt = probabilities.melt(
        id_vars=["oid"], var_name="class_name", value_name="probability"
    )
    
    probabilities_melt["ranking"] = (
        probabilities_melt.groupby("oid")["probability"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    probabilities_melt["sid"] = sid
    probabilities_melt["classifier_id"] = classifier_id
    probabilities_melt["classifier_version"] = classifier_version_str_to_small_integer(classifier_version)
    probabilities_melt["class_id"] = probabilities_melt["class_name"].map(
        lambda class_name: class_name_to_id(class_name, class_taxonomy)
    )

    oid_to_lastmjd = {oid: msg["jd"] - 2400000.5 for oid, msg in messages_dict.items()}
    probabilities_melt["lastmjd"] = probabilities_melt["oid"].map(oid_to_lastmjd)

    probabilities_melt = probabilities_melt.drop(columns=["class_name"])

    return probabilities_melt.to_dict(orient="records")


def get_taxonomy_by_classifier_id(classifier_id: int, psql_connection: PSQLConnection) -> dict[str, int]:
    """Fetch taxonomy from DB for a given classifier and return {class_name: class_id}.

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

def classifier_version_str_to_small_integer(version: str) -> int:
    """
    Convert a version string to a small integer.
    Example: "1.2.3" -> 123
    """
    parts = version.split(".")
    if len(parts) == 3:
        parts[-1] = parts[-1].split("_")[0]
        return int("".join(parts))
    return 0

def class_name_to_id(class_name: str, class_taxonomy: dict[str, int]) -> int:
    """
    Convert a class name to an integer ID.
    This is a placeholder function and should be replaced with actual logic.
    """

    class_dict = class_taxonomy
    return class_dict.get(class_name, -1)  # Return -1 if class_name not found