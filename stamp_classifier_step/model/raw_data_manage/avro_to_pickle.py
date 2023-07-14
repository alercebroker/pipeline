import pandas as pd
import fastavro
import pickle
import glob
import os
import sys
from tqdm import tqdm

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

from raw_data_manage.add_bogus_class import AddBogus
from raw_data_manage.alerce_api import AlerceStamps


class AvroConverter(object):
    def __init__(self, path_list, mnt_path, save_path, extra_features):
        self.path_list = pd.read_csv(path_list)
        self.mnt_path = mnt_path
        self.save_path = save_path
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.extra_features = extra_features

        self.n_objects = len(self.path_list)
        print(str(self.n_objects) + " total avro files")
        self.auxiliar_frame = self.path_list[["class", "oid"]]
        self.images_frame = pd.DataFrame(columns=self.stamp_keys + self.extra_features)
        self.collect_avros()
        self.merge_dataframes()
        file_pi = open(self.save_path, "wb")
        pickle.dump(self.auxiliar_frame, file_pi, protocol=2)

    def collect_avros(self):
        # if self.path_list is None:
        #     files = [f for f in glob.glob(self.mnt_path + "*.avro", recursive=False)]
        #     for i, p in enumerate(files):
        #         avro_data = self.read_avro(p)
        #         self.images_frame.loc[i] = list(avro_data.values())
        # else:
        #     for i in range(self.n_objects):
        #         avro_path = self.mnt_path+self.path_list.loc[i]["path"][1:]
        #         avro_data = self.read_avro(avro_path)
        #         self.images_frame.loc[i] = list(avro_data.values())
        for i in tqdm(range(self.n_objects)):
            avro_path = self.mnt_path + self.path_list.loc[i]["path"][1:]
            avro_data = self.read_avro(avro_path)
            self.images_frame.loc[i] = list(avro_data.values())

    def read_avro(self, avro_path):
        with open(avro_path, "rb") as f:
            freader = fastavro.reader(f)
            schema = freader.schema
            for i, packet in enumerate(freader):
                continue
            image_data = {}
            for key in self.stamp_keys:
                image_data[key] = packet[key]["stampData"]
            feature_data = {}
            for key in self.extra_features:
                # if key == "isdiffpos":
                #    isdiffpos = packet["candidate"][key]
                #    if isdiffpos == "t":
                #        feature_data[key] = 1
                #    else:
                #        feature_data[key] = 0
                # else:
                feature_data[key] = packet["candidate"][key]
            data = {**image_data, **feature_data}
        return data

    def merge_dataframes(self):
        self.auxiliar_frame = pd.concat(
            [self.auxiliar_frame, self.images_frame], axis=1
        )
        print(self.auxiliar_frame)


if __name__ == "__main__":
    # """
    # path_list = "/home/rcarrasco/stamp_classifier/oid_simplified_paths.csv"
    # mnt_path = "/home/rcarrasco/stamp_classifier/"
    # save_path = "/home/rcarrasco/stamp_classifier/pickles/alerts_for_training.pkl"
    #
    # converter = AvroConverter(path_list=path_list,
    #                           mnt_path=mnt_path,
    #                           save_path=save_path)
    # """
    # path_list = "/home/rcarrasco/Projects/stamp_classifier/pickles/tns_dataset"
    # save_path = "/home/rcarrasco/stamp_classifier/pickles/tns_sn.pkl"
    #
    # converter = AvroConverter(mnt_path=path_list,
    #                           save_path=save_path)
    path_list = "/home/rcarrasco/Projects/stamp_classifier/all_data/unique_oid.csv"
    mnt_path = "/home/rcarrasco/Projects/stamp_classifier/all_data/"
    save_path = (
        "/home/rcarrasco/Projects/stamp_classifier/pickles/training_set_extra_bogus.pkl"
    )

    features_to_add = [
        "sgscore1",
        "distpsnr1",
        "sgscore2",
        "distpsnr2",
        "sgscore3",
        "distpsnr3",
        "isdiffpos",
    ]

    converter = AvroConverter(
        path_list=path_list,
        mnt_path=mnt_path,
        save_path=save_path,
        extra_features=features_to_add,
    )

    data = pd.read_pickle(save_path)
    print("n_examples so far", len(data))
    # Adding bogus class to training set

    bogus_json_path = "/home/rcarrasco/Projects/ZTF_data/broker_bogus.json"
    training_set_path = os.path.join("../../pickles", "training_set_extra_bogus.pkl")
    save_path = os.path.join("../../pickles", "bogus_and_features.pkl")

    add_bogus = AddBogus(
        training_set_path=training_set_path,
        bogus_path=bogus_json_path,
        save_path=save_path,
        extra_features=features_to_add,
    )

    add_bogus.json2dataframe()
    add_bogus.append_to_training()
    print(add_bogus.final_frame)
    add_bogus.save_dataframe(overwrite=True)

    data = pd.read_pickle(training_set_path)
    print("n_examples so far", len(data))

    # Adding manually classified bogus using alerce api

    bogus_list = pd.read_hdf("../../../data_analysis/bogus_list.hdf", key="h")
    bogus_list = bogus_list["ZTF_id"].values
    alerce = AlerceStamps(save_path="../../all_data/data_from_api")
    alerce.download_avros(object_ids=bogus_list, file_name="alert_frame_list.pkl")
    alerce.create_frame_to_merge(
        frame_class="bogus",
        alerts_list="alert_frame_list.pkl",
        features=features_to_add,
    )
    alerce.append_to_training(training_set_path)
    alerce.save_dataframe(overwrite=True)

    data = pd.read_pickle(training_set_path)
    print("n_examples so far", len(data))
