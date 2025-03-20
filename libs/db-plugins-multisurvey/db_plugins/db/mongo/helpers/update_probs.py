from pymongo import UpdateOne
from pymongo.database import Database

"""
Helper function to create or update the probabilities for an object
"""


def get_probabilities(database: Database, oids: list):
    probabilities = database["object"].find(
        {"_id": {"$in": oids}}, {"probabilities": True}
    )
    return {item["_id"]: item["probabilities"] for item in probabilities}


def get_db_operations(
    classifier: str,
    version: str,
    oid: str,
    object_probabilities: list,
    probabilities: dict,
):
    """
    Check if this is really efficient
    """
    # sort by probabilities (for rank)
    sorted_classes_and_prob_list = sorted(
        probabilities.items(), key=lambda val: val[1], reverse=True
    )

    # Remove all existing probabilities for given classifier and version (if any)
    object_probabilities = [
        item
        for item in object_probabilities
        if item["classifier_name"] != classifier
        or item["classifier_version"] != version
    ]
    # Add all new probabilities
    object_probabilities.extend(
        [
            {
                "classifier_version": version,
                "classifier_name": classifier,
                "class_name": object_class,
                "probability": prob,
                "ranking": idx + 1,
            }
            for idx, (object_class, prob) in enumerate(sorted_classes_and_prob_list)
        ]
    )

    operation = UpdateOne(
        {"_id": oid}, {"$set": {"probabilities": object_probabilities}}
    )

    return operation


def create_or_update_probabilities(
    database: Database,
    classifier: str,
    version: str,
    oids: str,
    probabilities: dict,
):
    object_probs = get_probabilities(database, [oids])

    database["object"].bulk_write(
        [
            get_db_operations(
                classifier, version, oids, object_probs[oids], probabilities
            )
        ],
        ordered=False,
    )


def create_or_update_probabilities_bulk(
    database: Database,
    classifier: str,
    version: str,
    oids: list,
    probabilities: list,
):
    """
    Bulk update using the actual bulk object of pymongo
    """
    db_operations = []

    # no warrants that probs will have the same oid order
    object_probabilities = get_probabilities(database, oids)

    for oid, probs in zip(oids, probabilities):
        db_operations.append(
            get_db_operations(
                classifier, version, oid, object_probabilities[oid], probs
            )
        )

    database["object"].bulk_write(db_operations, ordered=False)
