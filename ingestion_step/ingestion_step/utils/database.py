import importlib.metadata
from typing import Any, List

from db_plugins.db.sql.models import Object
from sqlalchemy.dialects.postgresql import insert

from ..database import PsqlConnection

version = importlib.metadata.version("sorting-hat-step")


def insert_empty_objects_to_sql(db: PsqlConnection, records: List[dict[str, Any]]):
    # insert into db values = records on conflict do nothing
    def format_extra_fields(record: dict[str, Any]):
        extra_fields = record["extra_fields"]
        return {
            "ndethist": extra_fields["ndethist"],
            "ncovhist": extra_fields["ncovhist"],
            "mjdstarthist": extra_fields["jdstarthist"] - 2400000.5,
            "mjdendhist": extra_fields["jdendhist"] - 2400000.5,
            "meanra": record["ra"],
            "meandec": record["dec"],
            "firstmjd": record["mjd"],
            "lastmjd": record["mjd"],
            "deltajd": 0,
            "step_id_corr": version,
        }

    oids = {
        r["_id"]: format_extra_fields(r) for r in records if r["sid"].lower() == "ztf"
    }
    with db.session() as session:
        to_insert = [{"oid": oid, **extra_fields} for oid, extra_fields in oids.items()]
        statement = insert(Object).values(to_insert)
        statement = statement.on_conflict_do_update(
            "object_pkey",
            set_=dict(
                ndethist=statement.excluded.ndethist,
                ncovhist=statement.excluded.ncovhist,
                mjdstarthist=statement.excluded.mjdstarthist,
                mjdendhist=statement.excluded.mjdendhist,
            ),
        )
        session.execute(statement)
        session.commit()
