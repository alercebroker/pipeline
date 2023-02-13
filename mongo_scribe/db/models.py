from pymongo.collection import Collection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from db_plugins.db.mongo import MongoConnection
from mongo_scribe.command.exceptions import NonExistantCollectionException

models_dictionary = {
    "object": Object,
    "detection": Detection,
    "non_detection": NonDetection,
}


def get_model_collection(
    connection: MongoConnection, model_name: str
) -> Collection:
    """
    Returns the collection based on a model name string.
    Raises NonExistantCollection if the model name doesn't exist.
    """
    try:
        db_model = models_dictionary[model_name]
        return connection.query(db_model).collection
    except KeyError as exc:
        raise NonExistantCollectionException from exc
