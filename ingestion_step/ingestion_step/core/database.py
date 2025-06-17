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
    "sigmara",
    "sigmadec",
    "firstmjd",
    "lastmjd",
    "deltamjd",
    "n_det",
    "n_forced",
    "n_non_det",
    "corrected",
    "stellar",
]

DETECTION_COLUMNS = [
    "oid",
    "measurement_id",
    "mjd",
    "ra",
    "dec",
    "band",
]

FORCED_DETECTION_COLUMNS = [
    "oid",
    "measurement_id",
    "mjd",
    "ra",
    "dec",
    "band",
]
