from pymongo import MongoClient

DETECTION = "detection"
NON_DETECTION = "non_detection"
FORCED_PHOTOMETRY = "phorced_photometry"


class DatabaseConnection:
    def __init__(self, config: dict):
        self.config = config
        _database = self.config.pop("database")
        self.client = MongoClient(**self.config)
        self.database = self.client[_database]


def _get_mongo_detections(oids, db_mongo, parser) -> list:
    if db_mongo is None:
        return []
    db_detections = db_mongo.database[DETECTION].aggregate(
        [
            {"$match": {"oid": {"$in": oids}}},
            {
                "$addFields": {
                    "candid": {"$ifNull": ["$candid", {"$toString": "$_id"}]},
                    "forced": False,
                    "new": False,
                }
            },
            {
                "$project": {
                    "_id": False,
                    "evilDocDbHack": False,
                    "stellar": False,
                    "e_mag_corr": False,
                    "corrected": False,
                    "mag_corr": False,
                    "e_mag_corr_ext": False,
                    "dubious": False,
                }
            },
        ]
    )
    db_detections = parser(db_detections)
    return db_detections


def _get_mongo_non_detections(oids, db_mongo, parser):
    if db_mongo is None:
        return []
    db_non_detections = db_mongo.database[NON_DETECTION].find(
        {"oid": {"$in": oids}},
        {"_id": False, "evilDocDbHack": False},
    )
    db_non_detections = parser(db_non_detections)
    return db_non_detections


def _get_mongo_forced_photometries(oids, db_mongo, parser):
    if db_mongo is None:
        return []
    db_forced_photometries = db_mongo.database[FORCED_PHOTOMETRY].aggregate(
        [
            {"$match": {"oid": {"$in": oids}}},
            {
                "$addFields": {
                    "candid": "$_id",
                    "forced": True,
                    "new": False,
                }
            },
            {
                "$project": {
                    "_id": False,
                    "evilDocDbHack": False,
                    "stellar": False,
                    "e_mag_corr": False,
                    "corrected": False,
                    "mag_corr": False,
                    "e_mag_corr_ext": False,
                    "dubious": False,
                }
            },
        ]
    )
    db_forced_photometries = parser(db_forced_photometries)
    return db_forced_photometries
