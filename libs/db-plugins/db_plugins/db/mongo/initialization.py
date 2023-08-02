from ._connection import MongoConnection

# add import for models so that they can be created by the metadata
from .models import *


def init_mongo_database(config, db=None):
    db = db or MongoConnection(config)
    db.create_db()
