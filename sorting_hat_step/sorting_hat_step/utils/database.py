import math
from typing import Dict, Union, List

from sqlalchemy.dialects.postgresql import insert

from db_plugins.db.sql.models import Object
from ..database import MongoConnection, PsqlConnection
import importlib.metadata

version = importlib.metadata.version("sorting-hat-step")


def oid_query(db: MongoConnection, oid: list) -> Union[str, None]:
    """
    Query the database and check if any of the OIDs is already in database

    :param db: Database connection
    :param oid: oid of any survey

    :return: existing aid if exists else is None
    """
    found = db.database["object"].find_one({"_id": {"$in": oid}}, {"aid": 1})
    if found:
        return found["aid"]
    return None


def conesearch_query(
    db: MongoConnection, ra: float, dec: float, radius: float
) -> Union[str, None]:
    """
    Query the database and check if there is an alerce_id
    for the specified coordinates and search radius

    :param db: Database connection
    :param ra: first coordinate argument (RA)
    :param dec: first coordinate argument (Dec)
    :param radius: search radius (arcsec)

    :return: existing aid if exists else is None
    """
    found = db.database["object"].find_one(
        {
            "loc": {
                "$nearSphere": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [ra - 180, dec],
                    },
                    "$maxDistance": math.radians(radius / 3600) * 6.3781e6,
                },
            },
        },
        {"aid": 1},
    )
    if found:
        print(found)
        return found["aid"]
    return None


def update_query(db: MongoConnection, records: List[dict]):
    """
    Insert or update the records in a dictionary. Pushes the oid array to
    oid column.

    :param db: Database connection
    :param records: Records containing _id and oid fields to insert or update
    """
    for record in records:
        query = {"_id": record["_id"]}
        new_value = {
            "$set": {"aid": record["aid"]},
        }
        db.database["object"].find_one_and_update(
            query, new_value, upsert=True, return_document=True
        )


def insert_empty_objects_to_sql(db: PsqlConnection, records: List[Dict]):
    # insert into db values = records on conflict do nothing
    def format_extra_fields(record):
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
        r["_id"]: format_extra_fields(r)
        for r in records
        if r["sid"].lower() == "ztf"
    }
    with db.session() as session:
        to_insert = [
            {"oid": oid, **extra_fields} for oid, extra_fields in oids.items()
        ]
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
