import abc
import os
from typing import NamedTuple
from pprint import pprint
from pymongo.collection import Collection
from pymongo.operations import UpdateOne
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.mongo.models import Object, Detection, NonDetection
from mongo_scribe.command.exceptions import NonExistentCollectionException
from mongo_scribe.db.factories.update_probability import (
    UpdateProbabilitiesOperation,
)

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
    def update_probabilities(self, operation: UpdateProbabilitiesOperation):
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

    def update_probabilities(self, operation: UpdateProbabilitiesOperation):
        print("Updating probabilities")
        pprint(
            {"updates": operation.updates, "classifier": operation.classifier}
        )


class ScribeCollectionMongo(ScribeCollection):
    def __init__(self, connection: MongoConnection, collection_name):
        try:
            db_model = models_dictionary[collection_name]
            self.connection = connection
            self.collection: Collection = connection.query(db_model).collection
        except KeyError as exc:
            raise NonExistentCollectionException from exc

    def insert_many(self, inserts, ordered=False):
        if len(inserts) > 0:
            self.collection.insert_many(inserts, ordered=ordered)

    def bulk_write(self, updates, ordered=False):
        if len(updates) > 0:
            self.collection.bulk_write(updates, ordered=ordered)

    def update_probabilities(self, operation: UpdateProbabilitiesOperation):
        classifier = operation.classifier
        update_data = operation.updates
        if len(update_data) > 0:
            self.create_or_update_probabilities(
                classifier=classifier["classifier_name"],
                version=classifier["classifier_version"],
                probabilities=dict(update_data),
            )

    def _get_probabilities(self, aids: list):
        """Get probability list for all AIDs in given list from the database.

        Args:
            aids: List of AIDs to search for

        Returns:
            dict[str, list]: Mapping from AID to list of probabilities dictionaries
        """
        probabilities = self.collection.find(
            {"_id": {"$in": aids}}, {"probabilities": True}
        )
        probs = {aid: [] for aid in aids}  # This is the default
        # Existing ones will be updated
        probs.update(
            {item["_id"]: item["probabilities"] for item in probabilities}
        )
        return probs

    @classmethod
    def _create_probabilities(
        cls, classifier: str, version: str, probabilities: dict
    ):
        """Create new probabilities for extension of the existing one.

        Args:
            classifier: Name of the classifier to update
            version: Version of the classifier to update
            probabilities: Mapping from class to probability, e.g.,
                .. code-block:: python

                   {
                       "class1": 0.3,
                       "class2": 0.7,
                   }

        Returns:
            List of mappings of probabilities (as in DB), e.g.,
                .. code-block:: python

                   [
                       {
                           "classifier_name": "classifier",
                           "classifier_version": "1.0.0",
                           "class_name": "class2",
                           "probability": 0.7,
                           "ranking": 1
                       },
                       {
                           "classifier_name": "classifier",
                           "classifier_version": "1.0.0",
                           "class_name": "class1",
                           "probability": 0.3,
                           "ranking": 2
                       },
                   ]
        """
        probabilities = sorted(
            probabilities.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "classifier_version": version,
                "classifier_name": classifier,
                "class_name": class_,
                "probability": prob,
                "ranking": idx + 1,
            }
            for idx, (class_, prob) in enumerate(probabilities)
        ]

    @classmethod
    def _is_classifier_in(cls, classifier: str, probabilities: list):
        """Checks if given classifier exists within probabilities.

        Args:
            classifier: Name of the classifier to update
            probabilities: List of mappings of probabilities (as in DB), e.g.,
                .. code-block:: python

                   [
                       {
                           "classifier_name": "classifier",
                           "classifier_version": "1.0.0",
                           "class_name": "class1",
                           "probability": 0.5,
                           "ranking": 1
                       },
                       ...
                   ]

        Returns:
            bool: Whether the classifier already exists in the probabilities
        """
        return any(
            item["classifier_name"] == classifier for item in probabilities
        )

    @classmethod
    def _delete_classifier(cls, classifier: str, probabilities: list):
        """Delete all items with given classifier name from probabilities.

        Args:
            classifier: Name of the classifier to update
            probabilities: List of mappings of probabilities (as in DB), e.g.,
                .. code-block:: python

                   [
                       {
                           "classifier_name": "classifier",
                           "classifier_version": "1.0.0",
                           "class_name": "class1",
                           "probability": 0.5,
                           "ranking": 1
                       },
                       ...
                   ]

        Returns:
            list: Same structure as `probabilities` but with all items with the given classifier removed
        """
        return [
            item
            for item in probabilities
            if item["classifier_name"] != classifier
        ]

    @classmethod
    def _get_update_op(
        cls,
        classifier: str,
        version: str,
        old_probabilities: list,
        new_probabilities: dict,
        insert_only: bool = False,
    ):
        """Get necessary DB operations for probability update.

        Args:
            classifier: Name of the classifier to update
            version: Version of the classifier to update
            new_probabilities: Mapping from class to probability, e.g.,
                .. code-block:: python

                   {
                       "class1": 0.1,
                       "class2": 0.3,
                       ...
                   }

            old_probabilities: List of mappings of probabilities (as in DB), e.g.,
                .. code-block:: python

                   [
                       {
                           "classifier_name": "classifier",
                           "classifier_version": "1.0.0",
                           "class_name": "class1",
                           "probability": 0.5,
                           "ranking": 1
                       },
                       ...
                   ]

            insert_only: Whether to only insert if classifier already exists. Otherwise, update existing values

        Returns:
            dict: Update dictionary for MongoDB (empty if using `insert_only` and classifier already exists)
        """
        if insert_only and cls._is_classifier_in(
            classifier, old_probabilities
        ):
            return {}

        filtered_probabilities = cls._delete_classifier(
            classifier, old_probabilities
        )

        # Add all new probabilities
        filtered_probabilities.extend(
            cls._create_probabilities(classifier, version, new_probabilities)
        )

        return {"$set": {"probabilities": filtered_probabilities}}

    def create_or_update_probabilities(
        self,
        classifier: str,
        version: str,
        probabilities: dict,
    ):
        old = self._get_probabilities(list(probabilities))

        # Only as dictionaries
        updates = {
            aid: self._get_update_op(classifier, version, old[aid], p)
            for aid, p in probabilities.items()
        }
        # As proper UpdateOne class and filter empty updates
        updates = [
            UpdateOne({"_id": aid}, update)
            for aid, update in updates.items()
            if update
        ]

        self.collection.bulk_write(updates, ordered=False)


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
