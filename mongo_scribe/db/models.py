import os
from pprint import pprint
from pymongo.collection import Collection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo import MongoConnection
from mongo_scribe.command.exceptions import NonExistantCollectionException

models_dictionary = {
    "object": Object,
    "detection": Detection,
    "non_detection": NonDetection,
}


class CollectionMock:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        pass

    def insert_many(self, inserts, ordered=False):
        print(f"Inserting into {self.collection_name}:")
        pprint(inserts)

    def bulk_write(self, updates):
        print(f"Bulk writing into {self.collection_name}:")
        pprint(updates)


def get_model_collection(
    connection: MongoConnection, model_name: str
) -> Collection:
    """
    Returns the collection based on a model name string.
    Raises NonExistantCollection if the model name doesn't exist.
    """
    if os.getenv("MOCK_DB_COLLECTION", "False") == "True":
        return CollectionMock(model_name)

    try:
        db_model = models_dictionary[model_name]
        return connection.query(db_model).collection
    except KeyError as exc:
        raise NonExistantCollectionException from exc
