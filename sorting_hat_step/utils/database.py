from typing import Callable, List, Union
from db_plugins.db.mongo.connection import DatabaseConnection

from db_plugins.db.mongo.models import Object


def oid_query(db: DatabaseConnection) -> Callable[[list], Union[int, None]]:
    """
    Query to database if the oids has an alerce_id
    :param oid: oid of any survey
    :return: existing aid if exists else is None
    """

    def query(oid: list) -> Union[int, None]:
        mongo_query = db.query(Object)
        object = mongo_query.find_one({"oid": {"$in": oid}})
        if object:
            return object["aid"]
        return None

    return query


def conesearch_query(
    db: DatabaseConnection,
) -> Callable[[float, float, float], List[dict]]:
    def query(lon: float, lat: float, meter_radius: float) -> List[dict]:
        mongo_query = db.query(Object)
        cursor = mongo_query.collection.find(
            {
                "loc": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                        "$maxDistance": meter_radius,
                    }
                },
            },
            {"aid": 1},  # only return alerce_id
        )
        spatial = [i for i in cursor]
        return spatial

    return query
