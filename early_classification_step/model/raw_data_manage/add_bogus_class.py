import numpy as np
import json
import os
import base64
import pandas as pd
import pickle


class AddBogus(object):
    """
    Take bogus class from real bogus dataset
    Create a dataframe with bogus images
    Append the bogus dataframe to pickle training set and save a copy
    """

    def __init__(
        self,
        training_set_path,
        bogus_path,
        save_path,
        extra_features=[],
        bogus_class_name="bogus",
    ):
        self.training_set_path = training_set_path
        self.bogus_path = bogus_path
        self.bogus_alerts = json.load(open(bogus_path, "r"))["query_result"]
        self.bogus_class_name = bogus_class_name
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.extra_features = extra_features
        self.images_frame = pd.DataFrame(
            columns=["class", "oid"] + self.stamp_keys + extra_features
        )
        self.save_path = save_path

    def json2dataframe(self):

        for i, alert in enumerate(self.bogus_alerts):
            object_id = alert["objectId"]
            images = {}
            for key in self.stamp_keys:
                images[key] = base64.b64decode(
                    alert[key]["stampData"]["$binary"].encode()
                )

            feature_data = {}
            for key in self.extra_features:
                # if key == "isdiffpos":
                #    isdiffpos = alert["candidate"][key]
                #    if isdiffpos == "t":
                #        feature_data[key] = 1
                #    else:
                #        feature_data[key] = 0
                # else:
                feature_data[key] = alert["candidate"][key]
            self.images_frame.loc[i] = (
                [self.bogus_class_name, object_id]
                + list(images.values())
                + list(feature_data.values())
            )

    def append_to_training(self):
        aux_training_set = pd.read_pickle(self.training_set_path)
        self.final_frame = pd.concat(
            [aux_training_set, self.images_frame], ignore_index=True, axis=0
        )

    def save_dataframe(self, overwrite=False):
        if overwrite:
            pickle.dump(
                self.final_frame, open(self.training_set_path, "wb"), protocol=2
            )
        else:
            pickle.dump(self.final_frame, open(self.save_path, "wb"), protocol=2)


if __name__ == "__main__":
    bogus_json_path = "/home/rcarrasco/ZTF_data/broker_bogus.json"
    training_set_path = os.path.join("../../pickles", "corrected_oids_alerts.pkl")
    save_path = os.path.join("../../pickles", "training_set_with_bogus.pkl")

    features_to_add = [
        "sgscore1",
        "distpsnr1",
        "sgscore2",
        "distpsnr2",
        "sgscore3",
        "distpsnr3",
        "isdiffpos",
    ]

    add_bogus = AddBogus(
        training_set_path=training_set_path,
        bogus_path=bogus_json_path,
        save_path=save_path,
        extra_features=features_to_add,
    )

    add_bogus.json2dataframe()
    add_bogus.append_to_training()
    print(add_bogus.final_frame)
    add_bogus.save_dataframe()
