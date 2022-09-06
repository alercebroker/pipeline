from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo import models

"""
Helper function to create or update the probabilities for an object
"""


def create_or_update_probabilities(
    connection: MongoConnection,
    aid: str,
    classifier: str,
    version: str,
    probabilities: dict
):
    """
    Create or update the probabilities for the object wth aid. Usiing the MongoConnection
    The classes and probabilities lists must be the same length and map by index the probabilities for each class.
    params:
    connection: the mongo conection used to interface with the db
    aid: the identifier of the object
    classifier: the name of the classifier
    version: the version of the classifier
    probabilities: the probabilities dictionary the keys are the names and the values are the probabilitie
    """

    # helper filter function
    def filter_function(class_name):
        return (
            lambda ele: ele["classifier_name"] == classifier
            and ele["classifier_version"] == version
            and ele["class_name"] == class_name
        )

    # zip the lists
    classes_and_prob_list = list(probabilities.items())

    # sort by probabilities (for rank)
    sorted_classes_and_prob_list = sorted(classes_and_prob_list, key=lambda val: val[1], reverse=True)

    unmodified_object = connection.database["object"].find_one({"aid": aid})

    for indx in range(len(sorted_classes_and_prob_list)):
        founded = list(filter(
            filter_function(sorted_classes_and_prob_list[indx][0]),
            unmodified_object["probabilities"],
        ))

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
                            "class_name": sorted_classes_and_prob_list[indx][0],
                        }
                    },
                },
                {
                    "$set": {
                        "probabilities.$.probability": sorted_classes_and_prob_list[indx][1],
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
                            "class_name": sorted_classes_and_prob_list[indx][0],
                            "probability": sorted_classes_and_prob_list[indx][1],
                            "ranking": indx + 1,
                        }
                    }
                },
            )
