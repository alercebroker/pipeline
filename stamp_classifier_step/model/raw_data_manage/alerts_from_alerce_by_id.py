import pandas as pd
from pandas.io.json import json_normalize
from IPython.display import HTML
import os
import requests
from tqdm import tqdm
import pickle
import io
import fastavro
from fastavro import writer, reader, parse_schema
import numpy as np


class AlerceApi(object):
    def __init__(self, baseurl):
        self.baseurl = baseurl

    def query(self, params):
        # show api results
        r = requests.post(url="%s/query" % self.baseurl, json=params)
        df = pd.DataFrame(r.json())
        query_results = json_normalize(df.result)
        query_results.set_index("oid", inplace=True)
        return query_results

    def get_sql(self, params):
        r = requests.post(url="%s/get_sql" % self.baseurl, json=params)
        return r.content

    def get_detections(self, params):
        # show api results
        r = requests.post(url="%s/get_detections" % self.baseurl, json=params)
        df = pd.DataFrame(r.json())
        detections = json_normalize(df.result.detections)
        detections.set_index("candid", inplace=True)
        return detections

    def get_non_detections(self, params):
        # show api results
        r = requests.post(url="%s/get_non_detections" % self.baseurl, json=params)
        df = pd.DataFrame(r.json())
        detections = json_normalize(df.result.non_detections)
        detections.set_index("mjd", inplace=True)
        return detections

    def get_stats(self, params):
        # show api results
        r = requests.post(url="%s/get_stats" % self.baseurl, json=params)
        df = pd.DataFrame(r.json())
        stats = json_normalize(df.result.stats)
        stats.set_index("oid", inplace=True)
        return stats

    def get_probabilities(self, params):
        # show api results
        r = requests.post(url="%s/get_probabilities" % self.baseurl, json=params)
        early = json_normalize(r.json()["result"]["probabilities"]["early_classifier"])
        early.set_index("oid", inplace=True)
        late = json_normalize(r.json()["result"]["probabilities"]["random_forest"])
        late.set_index("oid", inplace=True)
        return early, late

    def get_features(self, params):
        # show api results
        r = requests.post(url="%s/get_features" % self.baseurl, json=params)
        features = json_normalize(r.json())
        features.set_index("oid", inplace=True)
        return features

    def plotstamp(self, oid, candid):
        science = (
            "http://avro.alerce.online/get_stamp?oid=%s&candid=%s&type=science&format=png"
            % (oid, candid)
        )
        images = """
        &emsp;&emsp;&emsp;&emsp;&emsp;
        Science
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 
        Template
        &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 
        Difference
        <div class="container">
        <div style="float:left;width:20%%"><img src="%s"></div>
        <div style="float:left;width:20%%"><img src="%s"></div>
        <div style="float:left;width:20%%"><img src="%s"></div>
        </div>
        """ % (
            science,
            science.replace("science", "template"),
            science.replace("science", "difference"),
        )
        display(HTML(images))


class AlerceStamps(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.alerce_url = "http://avro.alerce.online/get_avro"
        self.alerce = AlerceApi("http://ztf.alerce.online")
        self.image_types = ["science", "template", "difference"]
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]

    def _get_candid(self, object_ids):
        candids = []
        alert_series = []
        print("Downloading alert metadata")
        for ob_id in tqdm(object_ids, total=len(object_ids)):
            params = {"oid": ob_id}
            detections = self.alerce.get_detections(params)
            """if ob_id == "ZTF19acazaug":
                print(detections)

                print(detections["sigmagap_corr"].values[0])
                print(type(detections["sigmagap_corr"].values[0]))
                print(np.logical_not(np.isnan(detections["sigmagap_corr"].values)))
                print(np.logical_and(np.logical_or(detections["has_stamps"].values == True,
                                                  detections["has_stamps"].values == None),
                                                  np.logical_not(np.isnan(detections["sigmagap_corr"].values))))
                asdasd"""
            detections = detections[
                np.logical_and(
                    np.logical_or(
                        detections["has_stamps"].values == True,
                        detections["has_stamps"].values == None,
                    ),
                    np.logical_not(np.isnan(detections["sigmagap_corr"].values)),
                )
            ]
            detections.sort_values(by=["mjd"], inplace=True)
            alert_frame = detections.iloc[0]
            candid = alert_frame.candid_str
            candids.append(candid)
            alert_series.append(alert_frame)
        return candids, alert_series

    def change_savepath(self, save_path):
        self.save_path = save_path

    def download_avros(
        self,
        object_ids=None,
        file_name="alert_frame_list.pkl",
        retrun_alert_series=False,
        n_tries=10,
    ):
        if not os.path.exists(os.path.join(self.save_path, "avros/")):
            os.makedirs(os.path.join(self.save_path, "avros/"))
        if os.path.exists(os.path.join(self.save_path, file_name)):
            candids, alert_series = pd.read_pickle(
                os.path.join(self.save_path, file_name)
            )
        else:
            candids, alert_series = self._get_candid(object_ids)
            pickle.dump(
                [candids, alert_series],
                open(os.path.join(self.save_path, file_name), "wb"),
                protocol=2,
            )
        print("Downloading stamp data")
        for i, cand_id in tqdm(enumerate(candids), total=len(candids)):
            url = (
                self.alerce_url
                + "?oid="
                + alert_series[i].oid
                + "&candid="
                + str(cand_id)
            )
            save_path = os.path.join(
                self.save_path,
                "avros/",
                "alert_{}_{}.avro".format(alert_series[i].oid, cand_id),
            )
            if os.path.exists(save_path):
                continue
            for t in range(n_tries):
                resp = requests.get(url)
                avro = io.BytesIO(resp.content)
                try:
                    freader = fastavro.reader(avro)
                    break
                except:
                    print(save_path)
                    print("not a readable avro")
            schema = freader.schema
            packets = []
            for i, packet in enumerate(freader):
                packets.append(packet)
            with open(save_path, "wb") as out:
                writer(out, schema, packets)
        if retrun_alert_series:
            return alert_series

    def create_frame_to_merge(
        self, frame_class, alerts_list="alert_frame_list.pkl", features=[]
    ):
        _, alert_series = pd.read_pickle(os.path.join(self.save_path, alerts_list))
        # print(alert_series)
        columns = ["class", "oid"] + self.stamp_keys + features
        stamp_frame = pd.DataFrame(columns=columns)
        # print(alert_series)
        for serie in alert_series:
            image_data = {}
            # print(serie)
            image_data["class"] = frame_class
            image_data["oid"] = serie.oid
            cand_id = serie.name
            avro_path = os.path.join(
                self.save_path, "avros/", "alert_{}_{}.avro".format(serie.oid, cand_id)
            )
            if not os.path.exists(avro_path):
                print("alert_{}_{}.avro".format(serie.oid, cand_id), "does not exists")
                continue
            with open(avro_path, "rb") as f:
                freader = fastavro.reader(f)
                schema = freader.schema
                for i, packet in enumerate(freader):
                    continue
                for key in self.stamp_keys:
                    image_data[key] = packet[key]["stampData"]
                feature_data = {}
                for key in features:
                    # if key == "isdiffpos":
                    #    isdiffpos = packet["candidate"][key]
                    #    if isdiffpos == "t":
                    #        feature_data[key] = 1
                    #    else:
                    #        feature_data[key] = 0
                    # else:
                    feature_data[key] = packet["candidate"][key]
                data = {**image_data, **feature_data}
                stamp_frame = stamp_frame.append(data, ignore_index=True)
        self.frame_to_merge = stamp_frame
        return stamp_frame

    def append_to_training(self, training_set_path):
        self.training_set_path = training_set_path
        aux_training_set = pd.read_pickle(training_set_path)
        self.final_frame = pd.concat(
            [aux_training_set, self.frame_to_merge], ignore_index=True, axis=0
        )

    def save_dataframe(self, overwrite=False):
        if overwrite:
            pickle.dump(
                self.final_frame, open(self.training_set_path, "wb"), protocol=2
            )
        else:
            pickle.dump(self.final_frame, open(self.save_path, "wb"), protocol=2)


if __name__ == "__main__":
    bogus_list = pd.read_hdf("../../../data_analysis/bogus_list.hdf", key="h")
    bogus_list = bogus_list["ZTF_id"].values
    alerce = AlerceStamps("../../all_data/data_from_api")
    alerce.download_avros(object_ids=bogus_list, file_name="alert_frame_list.pkl")
    features_to_add = [
        "sgscore1",
        "distpsnr1",
        "sgscore2",
        "distpsnr2",
        "sgscore3",
        "distpsnr3",
        "isdiffpos",
    ]
    alerce.create_frame_to_merge(
        frame_class="bogus",
        alerts_list="alert_frame_list.pkl",
        features=features_to_add,
    )
