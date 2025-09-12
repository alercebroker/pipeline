from typing import Any, Hashable

from db_plugins.db.sql.models import (
    DeclarativeBase,
)
from sqlalchemy.dialects.postgresql import insert


def db_insert_on_conflict_do_nothing_builder(
    model: type[DeclarativeBase], data: list[dict[Hashable, Any]]
):
    stmt = insert(model).values(data)
    stmt = stmt.on_conflict_do_nothing()

    return stmt


def db_insert_on_conflict_do_update_builder(
    model: type[DeclarativeBase], data: list[dict[Hashable, Any]], pk: list[str]
):
    stmt = insert(model).values(data)
    stmt = stmt.on_conflict_do_update(
        constraint=model.__table__.primary_key.name,  # pyright: ignore
        set_={k: v for k, v in stmt.excluded.items() if k not in pk},
    )

    return stmt


OBJECT_COLUMNS = [
    "oid",
    "tid",
    "sid",
    "meanra",
    "meandec",
    "firstmjd",
    "lastmjd",
]

DETECTION_COLUMNS = [
    "oid",
    "sid",
    "measurement_id",
    "mjd",
    "ra",
    "dec",
    "band",
]

FORCED_DETECTION_COLUMNS = [
    "oid",
    "sid",
    "measurement_id",
    "mjd",
    "ra",
    "dec",
    "band",
]
