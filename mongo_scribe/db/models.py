from typing import Literal
from pymongo.collection import Collection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo import MongoConnection
from ..command.exceptions import NonExistantCollectionException

models_dictionary = {
    "object": Object,
    "detection": Detection,
    "non_detection": NonDetection
}

def get_model_collection(connection: MongoConnection, model_name: str) -> Collection:
    try:
        db_model = models_dictionary[model_name]
        return connection.query(db_model).collection
    except KeyError:
        raise NonExistantCollectionException