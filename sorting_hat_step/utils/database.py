import math
from typing import Union, List
from pymongo.errors import BulkWriteError

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

def id_query(db: DatabaseConnection, _ids: list) -> List[dict]:
    """
    Query the database and check which _id values are already in the database

    :param db: Database connection
    :param _ids: _id values to look for

    :return: All the found ids with the corresponding list of oids
    """
    mongo_query = db.query(Object)
    found_cursor = mongo_query.collection.find({"_id": {"$in": _ids}}, {"_id": 1, "oid": 1})
    return list(found_cursor)

def update_query(db: DatabaseConnection, records: List[dict]):
    mongo_query = db.query(Object)
    for record in records:
        query = {"_id": {"$in": [record["_id"]]}}
        new_value = { "$set": { 'oid': record['oid'] } }
        mongo_query.collection.update_one(query, new_value, upsert=True)
