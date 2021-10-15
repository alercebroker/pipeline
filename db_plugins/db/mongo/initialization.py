from db_plugins.db.mongo.connection import MongoConnection
from pymongo import GEOSPHERE, TEXT, HASHED


def init_mongo_database(config, db=None):
    db = db or MongoConnection()
    db.connect(config=config)
    db.create_db()


def init(DB_CONFIG, db=None):
    if "MONGO" in DB_CONFIG:
        db_config = DB_CONFIG["MONGO"]
        init_mongo_database(db_config, db)
