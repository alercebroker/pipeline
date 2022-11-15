from db_plugins.db.mongo.connection import MongoConnection
from pymongo import UpdateOne

"""
Helper function to create or update the probabilities for an object
"""


def _create_or_update_probabilities(
    connection: MongoConnection,
    classifier: str,
    version: str,
    aid: str,
    probabilities: dict,
):
    """
    Create or update the probabilities for the object wth aid. Usiing the MongoConnection
    The classes and probabilities lists must be the same length and map by index the probabilities for each class.
    params:
    connection: the mongo conection used to interface with the db
    aid: the identifier of the object
    classifier: the name of the classifier
    version: the version of the classifier
    probabilities: the probabilities dictionary the keys are the classes names and the values are the probabilitie
    """

    # helper filter function
    def filter_function(class_name):
        return (
            lambda ele: ele["classifier_name"] == classifier
            and ele["classifier_version"] == version
            and ele["class_name"] == class_name
        )

    # sort by probabilities (for rank)
    sorted_classes_and_prob_list = sorted(
        probabilities.items(), key=lambda val: val[1], reverse=True
    )

    unmodified_object = connection.database["object"].find_one({"aid": aid})
    for indx, (class_name, prob) in enumerate(sorted_classes_and_prob_list):
        founded = list(
            filter(
                filter_function(sorted_classes_and_prob_list[indx][0]),
                unmodified_object["probabilities"],
            )
        )

        # should be always 1?
        if len(founded):
            # update
            connection.database["object"].update_one(
                {
                    "aid": aid,
                    "probabilities": {
                        "$elemMatch": {
                            "classifier_name": classifier,
                            "classifier_version": version,
                            "class_name": class_name,
                        }
                    },
                },
                {
                    "$set": {
                        "probabilities.$.probability": prob,
                        "probabilities.$.ranking": indx + 1,
                    }
                },
            )
        else:
            # create
            connection.database["object"].update_one(
                {
                    "aid": aid,
                },
                {
                    "$push": {
                        "probabilities": {
                            "classifier_name": classifier,
                            "classifier_version": version,
                            "class_name": class_name,
                            "probability": prob,
                            "ranking": indx + 1,
                        }
                    }
                },
            )


def _create_or_update_probabilities_bulk(
    connection: MongoConnection,
    classifier: str,
    version: str,
    aids: list,
    probabilities: list,
):
    """
    Bulk version of the create or update probabilities. It iterate over the list arguments and
    call create_or_update_probabilities
    params:
    connection: the mongo conection used to interface with the db
    classifier: the name of the classifier
    version: the version of the classifier
    aids: list of identifiers for the objects. It's index must be correlated with the probabilities list.
    probabilities: List of dicts with classes and probabilities. probabilities [i] should be for aid[i]
    """

    for indx in range(len(aids)):
        _create_or_update_probabilities(
            connection, classifier, version, aids[indx], probabilities[indx]
        )


def get_db_operations(classifier: str, version: str, aid: str, probabilities: dict):
    """
    Check if this is really efficient
    """

    db_operations = []

    sorted_classes_and_prob_list = sorted(
        probabilities.items(), key=lambda val: val[1], reverse=True
    )

    for n_item, (class_name, prob) in enumerate(sorted_classes_and_prob_list):
        db_operations.append(
            UpdateOne(
                {
                    "aid": aid,
                    "probabilities": {
                        "$elemMatch": {
                            "classifier_name": classifier,
                            "classifier_version": version,
                            "class_name": class_name,
                        }
                    },
                },
                {
                    "$set": {
                        "probabilities.$.probability": prob,
                        "probabilities.$.ranking": n_item + 1,
                    }
                },
            )
        )

    return db_operations


def create_or_update_probabilities_bulk(
    connection: MongoConnection,
    classifier: str,
    version: str,
    aids: list,
    probabilities: list,
):
    """
    Bulk update using the actual bulk object of pymongo
    """
    db_operations = []

    for aid, probs in zip(aids, probabilities):
        db_operations.extend(get_db_operations(classifier, version, aid, probs))

    connection.database["object"].bulk_write(db_operations)
