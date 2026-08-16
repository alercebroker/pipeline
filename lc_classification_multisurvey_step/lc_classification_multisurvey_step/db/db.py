"""Read-only database access for the multisurvey LC classification step.

Two queries, both run once at startup: classifier names -> ids, and classifier
ids -> {class_name: class_id}. The step never writes to the database; the scribe
owns the probability upsert (design doc §2, decision 3).

Unlike stamp_classifier_2025_multisurvey_step's reader, neither query swallows
exceptions: an unreachable database must not look like an unseeded table.
"""
import logging
from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

log = logging.getLogger(__name__)


def get_db_url(config: dict) -> str:
    return (
        f"postgresql://{config['USER']}:{config['PASSWORD']}"
        f"@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
    )


class PSQLConnection:
    """Session factory over a psql engine, scoped to `SCHEMA` via search_path.

    Copied from stamp_classifier_2025_multisurvey_step/.../db/db.py so the two
    multisurvey classifier steps connect identically.
    """

    def __init__(self, db_config: dict, engine=None, poolclass: str | None = None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)
        pool = NullPool if poolclass == "NullPool" else None

        if schema:
            self._engine = engine or create_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
                poolclass=pool,
            )
        else:
            self._engine = engine or create_engine(db_url, echo=False, poolclass=pool)

        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def get_classifier_ids_by_name(classifier_names: list, psql_connection) -> dict:
    """{classifier_name: {"classifier_id": int, "classifier_version": str}}.

    The `classifier` table's primary key is `classifier_id` alone, so a duplicated
    `classifier_name` is possible and is a deploy error. This is the only place it
    is still visible (the return value is keyed by name), so it raises here —
    design doc §8, assertion 2.
    """
    statement = text(
        "SELECT classifier_id, classifier_name, classifier_version "
        "FROM classifier WHERE classifier_name IN :names"
    ).bindparams(bindparam("names", expanding=True))

    found: dict = {}
    duplicates: set = set()
    with psql_connection.session() as session:
        for row in session.execute(statement, {"names": list(classifier_names)}).mappings():
            name = row["classifier_name"]
            if name in found:
                duplicates.add(name)
            found[name] = {
                "classifier_id": int(row["classifier_id"]),
                "classifier_version": row["classifier_version"],
            }

    if duplicates:
        raise ValueError(
            f"classifier table has more than one row for {sorted(duplicates)}; "
            "cannot resolve a classifier_id unambiguously"
        )
    return found


def get_taxonomy_by_classifier_id(classifier_ids: list, psql_connection) -> dict:
    """{classifier_id: {class_name: class_id}} from the taxonomy table.

    Ordered by "order" per classifier — cosmetic for the dict, kept to match the
    offline reference and the stamp step.
    """
    statement = text(
        "SELECT classifier_id, class_id, class_name FROM taxonomy "
        'WHERE classifier_id IN :classifier_ids ORDER BY classifier_id, "order"'
    ).bindparams(bindparam("classifier_ids", expanding=True))

    maps: dict = {}
    with psql_connection.session() as session:
        rows = session.execute(statement, {"classifier_ids": list(classifier_ids)})
        for row in rows.mappings():
            maps.setdefault(int(row["classifier_id"]), {})[row["class_name"]] = int(
                row["class_id"]
            )
    return maps
