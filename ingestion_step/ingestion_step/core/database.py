from typing import Any, Hashable, Iterable

from db_plugins.db.sql.models import (
    DeclarativeBase,
)
from sqlalchemy.dialects.postgresql import insert


def db_statement_builder(
    model: type[DeclarativeBase],
    data: list[dict[Hashable, Any]],
    conflict_columns: Iterable[Any] | None = None,
):
    stmt = insert(model).values(data)
    if conflict_columns:
        stmt = stmt.on_conflict_do_nothing(index_elements=conflict_columns)
    else:
        stmt = stmt.on_conflict_do_nothing()
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
