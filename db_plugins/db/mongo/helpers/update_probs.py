from db_plugins.db.mongo.connection import MongoConnection
from pymongo import UpdateOne

"""
Helper function to create or update the probabilities for an object
"""


def get_db_operations(
    connection: MongoConnection,
    classifier: str,
    version: str,
    aid: str,
    probabilities: dict,
):
    """
    Check if this is really efficient
    """

    db_operations = []

    sorted_classes_and_prob_list = sorted(
        probabilities.items(), key=lambda val: val[1], reverse=True
    )

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
    for n_item, (class_name, prob) in enumerate(sorted_classes_and_prob_list):
        found = list(
            filter(
                filter_function(sorted_classes_and_prob_list[n_item][0]),
                unmodified_object["probabilities"],
            )
        )
        if len(found):
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
        else:
            db_operations.append(
                UpdateOne(
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
                                "ranking": n_item + 1,
                            }
                        }
                    },
                )
            )

    return db_operations


def create_or_update_probabilities(
    connection: MongoConnection,
    classifier: str,
    version: str,
    aid: str,
    probabilities: dict,
):
    connection.database["object"].bulk_write(
        get_db_operations(connection, classifier, version, aid, probabilities),
        ordered=False,
    )


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
        db_operations.extend(
            get_db_operations(connection, classifier, version, aid, probs)
        )

    connection.database["object"].bulk_write(db_operations, ordered=False)
