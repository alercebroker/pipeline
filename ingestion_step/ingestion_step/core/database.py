from typing import Any, Hashable

from db_plugins.db.sql.models import (
    DeclarativeBase,
)
from sqlalchemy.dialects.postgresql import insert


def db_statement_builder(model: type[DeclarativeBase], data: list[dict[Hashable, Any]]):
    stmt = insert(model).values(data).on_conflict_do_nothing()
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
