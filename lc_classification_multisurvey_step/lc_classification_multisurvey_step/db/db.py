"""Read-only database access for the multisurvey LC classification step.

Two queries, both run once at startup by `resolve_classifiers`, which also
enforces the §8 fail-fast assertions: classifier names -> ids, and classifier
ids -> {class_name: class_id}. The step never writes; the scribe owns the
probability upsert (design doc §2, decision 3).

Unlike the stamp step's reader, neither query swallows exceptions: an
unreachable database must not look like an unseeded table.
"""
import logging
from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import URL, bindparam, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from ..probabilities import classifier_version_to_smallint

log = logging.getLogger(__name__)


def get_db_url(config: dict) -> URL:
    """Build the connection URL, escaping the credentials.

    `URL.create`, not the f-string the sibling steps use: a password has no
    constrained alphabet, and an `@` in it ends the userinfo early so psycopg2
    resolves the rest as the host (`/`, `:`, `?`, `#` corrupt it similarly).
    `URL.create` escapes each component and never renders the password in repr.
    """
    return URL.create(
        drivername="postgresql",
        username=config["USER"],
        password=config["PASSWORD"],
        host=config["HOST"],
        port=int(config["PORT"]),
        database=config["DB_NAME"],
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
            "cannot resolve a classifier_id unambiguously. Refusing to start."
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


def resolve_classifiers(classifier_names: list, model_version: str, psql_connection):
    """Resolve the head names to ids and fetch their taxonomy, or refuse to start.

    Returns ({classifier_name: classifier_id}, {classifier_id: {class_name: class_id}}).

    The five §8 startup assertions, all fatal: an unparseable, unseeded,
    partially-seeded, ambiguous or version-skewed classifier/taxonomy is a deploy
    error, and a step that started anyway would drop every probability or write
    it against the wrong classifier.

      1. MODEL_VERSION parses to a non-zero version smallint (here)
      2. every head name resolved to a row          (here)
      3. no name resolved to more than one row      (get_classifier_ids_by_name)
      4. every resolved id has a non-empty taxonomy (here)
      5. each row's classifier_version == model_version (here)
    """
    if classifier_version_to_smallint(model_version) == 0:
        raise ValueError(
            f"MODEL_VERSION '{model_version}' does not parse to a version smallint "
            "(expected three dot-separated parts, e.g. '2.1.0'). Every probability "
            "row would be written with classifier_version=0. Refusing to start."
        )

    found = get_classifier_ids_by_name(classifier_names, psql_connection)

    missing = [name for name in classifier_names if name not in found]
    if missing:
        raise ValueError(
            f"classifier table has no row for {missing}; the classifier seed "
            "is missing or incomplete in this schema. Refusing to start."
        )

    skewed = {
        name: row["classifier_version"]
        for name, row in found.items()
        if row["classifier_version"] != model_version
    }
    if skewed:
        raise ValueError(
            f"classifier.classifier_version {skewed} does not match MODEL_VERSION "
            f"'{model_version}'; the seeded taxonomy may not match the model's "
            "classes_. Refusing to start."
        )

    classifier_ids = {name: found[name]["classifier_id"] for name in classifier_names}
    taxonomy_maps = get_taxonomy_by_classifier_id(list(classifier_ids.values()), psql_connection)

    unseeded = {
        name: cid for name, cid in classifier_ids.items() if not taxonomy_maps.get(cid)
    }
    if unseeded:
        raise ValueError(
            f"taxonomy table has no rows for {unseeded} (name -> classifier_id); "
            "every probability for those heads would be dropped. Refusing to start."
        )

    log.info("resolved classifier ids %s", classifier_ids)
    return classifier_ids, taxonomy_maps
