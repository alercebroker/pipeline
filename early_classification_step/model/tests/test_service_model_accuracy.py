import io
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

from deployment.stamp_clf import StampClassifier
from parameters import general_keys, param_keys
from modules.data_set_generic import Dataset
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
import warnings

IP = "localhost"
N_SAMPLES_TO_TEST = -1


# IP = '18.191.43.15'


class TestCLFFeaturesMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data_path = os.path.join(
            PROJECT_PATH, "../pickles", "training_set_May-06-2020.pkl"
        )
        self.service_predictions_dict = self._get_service_predictions_dict(
            self, samples_to_get=N_SAMPLES_TO_TEST
        )
        self.dataset_preprocessed = self._get_dataset_to_process_by_local_model(self)
        self.model = self._get_pretrained_local_model(self)
        self.class_names = np.array(["AGN", "SN", "VS", "asteroid", "bogus"])
        self.class_dict = dict(
            zip(self.class_names, list(range(len(self.class_names))))
        )

    def _get_dataset_to_process_by_local_model(self):
        aux_model = StampClassifier()
        aux_model.data_loader.data_path = self.data_path
        aux_model.data_loader.set_dumping_data_to_pickle(False)
        aux_model.data_loader.features_names_list.append("oid")
        datasets_dict_with_oids = (
            aux_model.data_loader.get_preprocessed_datasets_splitted()
        )
        images = np.concatenate(
            [
                datasets_dict_with_oids[general_keys.TRAIN].data_array,
                datasets_dict_with_oids[general_keys.VALIDATION].data_array,
                datasets_dict_with_oids[general_keys.TEST].data_array,
            ],
            axis=0,
        )
        labels = np.concatenate(
            [
                datasets_dict_with_oids[general_keys.TRAIN].data_label,
                datasets_dict_with_oids[general_keys.VALIDATION].data_label,
                datasets_dict_with_oids[general_keys.TEST].data_label,
            ],
            axis=0,
        )
        metadata = np.concatenate(
            [
                datasets_dict_with_oids[general_keys.TRAIN].meta_data,
                datasets_dict_with_oids[general_keys.VALIDATION].meta_data,
                datasets_dict_with_oids[general_keys.TEST].meta_data,
            ],
            axis=0,
        )
        dataset = Dataset(
            data_array=images, data_label=labels, batch_size=None, meta_data=metadata
        )
        aux_model.close()
        del aux_model
        return dataset

    def _get_service_predictions_dict(self, samples_to_get=-1):
        service_predictions_dict = dict()
        input_data = pd.read_pickle(self.data_path)
        # print(input_data.columns)
        for col_name in [
            "ecl_lat",
            "ecl_long",
            "gal_lat",
            "gal_long",
            "non_detections",
        ]:
            del input_data[col_name]
        # print(input_data.columns)
        # print(input_data)
        input_data = input_data.sample(frac=1).reset_index(drop=True)
        # print(input_data)
        for i, test in tqdm(input_data.iterrows()):
            if i == samples_to_get:
                return service_predictions_dict
            metadata_columns = [
                c
                for c in test.keys()
                if c not in ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
            ]
            files = {
                "cutoutScience": io.BytesIO(test["cutoutScience"]),
                "cutoutTemplate": io.BytesIO(test["cutoutTemplate"]),
                "cutoutDifference": io.BytesIO(test["cutoutDifference"]),
            }

            metadata = io.StringIO()
            metadata_df = pd.DataFrame(test[metadata_columns]).transpose().copy()
            oid = metadata_df["oid"].values[0]
            # print(oid)
            metadata_df.set_index("oid", inplace=True)
            metadata_df.to_csv(metadata)
            files["metadata"] = metadata.getvalue()

            resp = requests.post(f"http://{IP}:5000/get_classification", files=files)
            if resp.status_code == 200:
                probs = resp.content
                probs = json.loads(probs)
                # print(probs)
                service_predictions_dict[oid] = probs
        return service_predictions_dict

    def _get_pretrained_local_model(self):
        params = {
            param_keys.NUMBER_OF_CLASSES: 5,
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: None,
            param_keys.BATCHNORM_CONV: None,
            param_keys.DROP_RATE: 0.5,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.NANS_TO: 0,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.CROP_SIZE: 21,
            param_keys.ENTROPY_REG_BETA: 0.5,
            param_keys.BATCH_SIZE: 64,
            param_keys.FEATURES_NAMES_LIST: [
                "sgscore1",
                "distpsnr1",
                "sgscore2",
                "distpsnr2",
                "sgscore3",
                "distpsnr3",
                "isdiffpos",
                "fwhm",
                "magpsf",
                "sigmapsf",
                "ra",
                "dec",
                "diffmaglim",
                "rb",
                "distnr",
                "magnr",
                "classtar",
                "ndethist",
                "ncovhist",
                "ecl_lat",
                "ecl_long",
                "gal_lat",
                "gal_long",
                "non_detections",
                "chinr",
                "sharpnr",
            ],
            param_keys.BATCHNORM_FEATURES_FC: True,
            param_keys.FEATURES_CLIPPING_DICT: {
                "sgscore1": [-1, "max"],
                "distpsnr1": [-1, "max"],
                "sgscore2": [-1, "max"],
                "distpsnr2": [-1, "max"],
                "sgscore3": [-1, "max"],
                "distpsnr3": [-1, "max"],
                "fwhm": ["min", 10],
                "distnr": [-1, "max"],
                "magnr": [-1, "max"],
                "ndethist": ["min", 20],
                "ncovhist": ["min", 3000],
                "chinr": [-1, 15],
                "sharpnr": [-1, 1.5],
                "non_detections": ["min", 2000],
            },
        }
        model = DeepHiTSWithFeaturesEntropyReg(params)
        model_path = os.path.join(PROJECT_PATH, "results/best_model_so_far")
        model.load_model_and_feature_stats(model_path)
        return model

    def _service_pred_dict_to_array(self, service_prediction_dict):
        predictions = np.zeros(len(self.class_names))
        for key in service_prediction_dict.keys():
            predictions[self.class_dict[key]] = service_prediction_dict[key]
        return predictions

    def _get_images_feature_class_from_dataset(self, oid):
        all_oids = self.dataset_preprocessed.meta_data[:, -1]
        oid_idx = np.argwhere(all_oids == oid)[0][0]
        images = self.dataset_preprocessed.data_array[oid_idx][None, ...]
        features = self.dataset_preprocessed.meta_data[oid_idx, :-1][None, ...]
        label = self.dataset_preprocessed.data_label[oid_idx]
        return images, features, label

    def test_local_predictions_match_service(self):
        predictions = []
        labels = []
        n_disagreements = 0
        for oid in tqdm(self.service_predictions_dict):
            service_pred_dict = self.service_predictions_dict[oid]
            service_pred_array = self._service_pred_dict_to_array(service_pred_dict)
            images, features, label_i = self._get_images_feature_class_from_dataset(oid)
            local_model_pred = self.model.predict_proba(
                data_array=images, features=features
            )[0]
            agreement_i = np.mean(local_model_pred == service_pred_array)
            if agreement_i != 1.0:
                warnings.warn(
                    "oid %s, agreement %s \n local_pred %s \n server_pred %s"
                    % (
                        oid,
                        str(agreement_i),
                        str(local_model_pred),
                        str(service_pred_array),
                    )
                )
                n_disagreements += 1
            self.assertEqual(
                np.argmax(local_model_pred),
                np.argmax(service_pred_array),
                msg="oid %s, missmatch predicted class\n local_pred %s; server_pred %s"
                % (
                    oid,
                    str(np.argmax(local_model_pred)),
                    str(np.argmax(service_pred_array)),
                ),
            )
        predictions.append(local_model_pred)
        labels.append(label_i)
        predictions = np.array(predictions)
        labels = np.array(labels)
        pred_classes = np.argmax(predictions, axis=-1)
        # print(labels)
        # print(pred_classes)
        print(
            "Accuracy of model %s, N_disagreements %i"
            % (str(np.mean(labels == pred_classes)), n_disagreements)
        )


if __name__ == "__main__":
    unittest.main()
