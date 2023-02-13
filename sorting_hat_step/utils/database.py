from typing import Callable, Dict, List, Union
from db_plugins.db.mongo.connection import DatabaseConnection

from db_plugins.db.mongo.models import Object


def db_queries(db: DatabaseConnection) -> Dict[str, Callable]:
    """
    Get a dictionary of functions that perform database queries with
    the db connection instance injected into them.

    :param db: instance of db_plugins.DatabaseConnection

    :return: dictionary with oid query and conesearch query
    """

    def oid_query(oid: list) -> Union[int, None]:
        """
        Query the database and check if the oids has an alerce_id

        :param oid: oid of any survey

        :return: existing aid if exists else is None
        """
        mongo_query = db.query(Object)
        object = mongo_query.find_one({"oid": {"$in": oid}})
        if object:
            return object["aid"]
        return None

    def conesearch_query(lon: float, lat: float, meter_radius: float) -> List[dict]:
        """
        Query the database and check if there is an alerce_id
        for the specified coordinates and search radius

        :param lon: first coordinate argument (RA)
        :param lat: first coordinate argument (Dec)
        :param meter_radius: search radius (arcsec)

        :return: existing aid if exists else is None
        """
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

    return {"oid_query": oid_query, "conesearch_query": conesearch_query}
