import abc
import os
from typing import NamedTuple
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


class Classifier(NamedTuple):
    classifier_name: str
    classifier_version: str


class ScribeCollection(abc.ABC):
    @abc.abstractmethod
    def insert_many(self, inserts, ordered=False):
        ...

    @abc.abstractmethod
    def bulk_write(self, updates):
        ...

    @abc.abstractmethod
    def update_probabilities(self, classifier: Classifier, update_data: list):
        ...


class ScribeCollectionMock(ScribeCollection):
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def insert_many(self, inserts, ordered=False):
        print(f"Inserting into {self.collection_name}:")
        pprint(inserts)

    def bulk_write(self, updates):
        print(f"Bulk writing into {self.collection_name}:")
        pprint(updates)

    def update_probabilities(self, classifier: Classifier, update_data: list):
        print("Updating probabilities")
        pprint(classifier, update_data)


class ScribeCollectionMongo(ScribeCollection):
    def __init__(self, connection: MongoConnection, collection_name):
        try:
            db_model = models_dictionary[collection_name]
            self.connection = connection
            self.collection: Collection = connection.query(db_model).collection
        except KeyError as exc:
            raise NonExistantCollectionException from exc

    def insert_many(self, inserts, ordered=False):
        if len(inserts) > 0:
            self.collection.insert_many(inserts, ordered=ordered)

    def bulk_write(self, updates):
        if len(updates) > 0:
            self.collection.bulk_write(updates)

    def update_probabilities(self, classifier: Classifier, update_data: list):
        ...


def get_model_collection(
    connection: MongoConnection, model_name: str
) -> ScribeCollection:
    """
    Returns the collection based on a model name string.
    Raises NonExistantCollection if the model name doesn't exist.
    """
    if os.getenv("MOCK_DB_COLLECTION", "False") == "True":
        return ScribeCollectionMock(model_name)

    return ScribeCollectionMongo(connection, model_name)
