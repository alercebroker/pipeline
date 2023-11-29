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


def _get_mongo_detections(aids, db_mongo):
    db_detections = db_mongo.database[DETECTION].aggregate(
        [
            {"$match": {"aid": {"$in": aids}}},
            {
                "$addFields": {
                    "candid": "$_id",
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
    return db_detections


def _get_mongo_non_detections(aids, db_mongo):
    db_non_detections = db_mongo.database[NON_DETECTION].find(
        {"aid": {"$in": aids}},
        {"_id": False, "evilDocDbHack": False},
    )
    return db_non_detections


def _get_mongo_forced_photometries(aids, db_mongo):
    db_forced_photometries = db_mongo.database[FORCED_PHOTOMETRY].aggregate(
        [
            {"$match": {"aid": {"$in": aids}}},
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
    return db_forced_photometries
