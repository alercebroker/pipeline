from sqlalchemy.dialects.postgresql import insert
from db_plugins.db.sql.models import Probability
from contextlib import contextmanager
from typing import Callable, ContextManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
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
    classifier_id: int,
    classifier_version: str,
    predictions: list[dict],
) -> None:
    if len(predictions) == 0:
        return

    with psql_connection.session() as session:
        data = _format_data(classifier_id, classifier_version, predictions)

        insert_stmt = insert(Probability)
        insert_stmt = insert_stmt.on_conflict_do_nothing()

        session.execute(insert_stmt, data)
        session.commit()


def _format_data(
    classifier_id: int, classifier_version: str, predictions: list[dict]
) -> list[dict]:
    logging.warning("No clue what LSST's sid is, setting to 0")
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
                    "class_id": class_name_to_id(class_name),
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
# CLASS_DICT = {"SN":0, "AGN":1,"VS":2, "asteroid":3, "bogus":4, "satellite":5}
CLASS_DICT = {"AGN":0, "SN":1,"VS":2, "asteroid":3, "bogus":4}


def class_name_to_id(class_name: str) -> int:
    """
    Convert a class name to an integer ID.
    This is a placeholder function and should be replaced with actual logic.
    """

    class_dict = CLASS_DICT#{name: idx for idx, name in enumerate(CLASS_LIST)}
    return class_dict.get(class_name, -1)  # Return -1 if class_name not found


# TODO: The following function is a placeholder and should be replaced with the actual implementation
def class_id_to_name(class_id: int) -> str:
    """
    Convert a class ID to a class name.
    This is a placeholder function and should be replaced with actual logic.
    """
    class_dict = {idx: name for name, idx in CLASS_DICT.items()}
    return class_dict.get(class_id, "unknown")  # Return "unknown" if class_id not found
