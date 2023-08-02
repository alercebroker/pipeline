from ._connection import MongoConnection


def init_mongo_database(config, db=None):
    db = db or MongoConnection()
    db.create_db()
