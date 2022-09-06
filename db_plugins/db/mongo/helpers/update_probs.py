from db_plugins.db.mongo.connection import MongoConnection
from db_plugins.db.mongo import models

"""
Helper function to create or update the probabilities for an object
"""


def create_of_update_probabilities(
    connection: MongoConnection,
    aid: str,
    classifier: str,
    version: str,
    classes: list,
    probabilities: list,
):
    """
    Create or update the probabilities for the object wth aid. Usiing the MongoConnection
    The classes and probabilities lists must be the same length and map by index the probabilities for each class.
    params:
    connection: the mongo conection used to interface with the db
    aid: the identifier of the object
    classifier: the name of the classifier
    version: the version of the classifier
    classes: the list of classes that the classifier returns
    probabilities: the probabilities for each of the classes the classifier returns
    """

    # helper filter function
    def filter_function(class_name):
        return (
            lambda ele: ele["classifier_name"] == classifier
            and ele["classifier_version"] == version
            and ele["class_name"] == class_name
        )

    if len(classes) != len(probabilities):
        raise Exception("The claases and probablities lists dont match size")

    # zip the lists
    zipped_classes_and_probs = list(zip(classes, probabilities))

    # sort by probabilities (for rank)
    sorted_zipped_list = sorted(zipped_classes_and_probs, key=lambda val: val[1], reverse=True)
    print(f"list : {sorted_zipped_list}")

    unmodified_object = connection.database["object"].find_one({"aid": aid})

    for indx in range(len(sorted_zipped_list)):
        founded = list(filter(
            filter_function(sorted_zipped_list[indx][0]),
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
                            "class_name": sorted_zipped_list[indx][0],
                        }
                    },
                },
                {
                    "$set": {
                        "probabilities.$.probability": sorted_zipped_list[indx][1],
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
                            "class_name": sorted_zipped_list[indx][0],
                            "probability": sorted_zipped_list[indx][1],
                            "ranking": indx + 1,
                        }
                    }
                },
            )
