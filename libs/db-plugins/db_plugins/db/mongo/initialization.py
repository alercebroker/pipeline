from .connection import MongoConnection


def init_mongo_database(config, db=None):
    db = db or MongoConnection()
    db.connect(config=config)
    db.create_db()
