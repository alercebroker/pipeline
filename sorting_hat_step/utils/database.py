import math
from typing import Union, List

from db_plugins.db.mongo.connection import DatabaseConnection
from db_plugins.db.mongo.models import Object


def oid_query(db: DatabaseConnection, oid: list) -> Union[str, None]:
    """
    Query the database and check if any of the OIDs is already in database

    :param db: Database connection
    :param oid: oid of any survey

    :return: existing aid if exists else is None
    """
    mongo_query = db.query(Object)
    found = mongo_query.collection.find_one({"oid": {"$in": oid}}, {"_id": 1})
    if found:
        return found["_id"]
    return None


def conesearch_query(
    db: DatabaseConnection, ra: float, dec: float, radius: float
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
    mongo_query = db.query(Object)
    found = mongo_query.collection.find_one(
        {
            "loc": {
                "$nearSphere": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [ra - 180, dec],
                    },
                    "$maxDistance": math.radians(radius / 3600),
                },
            },
        },
        {"_id": 1},
    )
    if found:
        return found["_id"]
    return None


def insert_query(db: DatabaseConnection, records: List[dict]):
    mongo_query = db.query(Object)
    result = mongo_query.collection.insert_many(records)
    return result.inserted_ids
