import os
import pickle
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
import numpy as np
from parameters import general_keys, param_keys
from modules.data_loaders.frame_to_input import get_image_from_bytes_stamp
from modules.data_set_generic import Dataset
import logging
import ephem
import pandas as pd


class StampClassifier(DeepHiTSWithFeaturesEntropyReg):
    def __init__(self):
        super().__init__(params={}, model_name="StampClassifier", session=None)
        model_path = os.path.join(
            PROJECT_PATH,
            "results/staging_model/DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759",
        )
        self.load_model_and_feature_stats(model_path)

    def _update_new_default_params(self):
        new_default_params = {
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
                "ndethist": ["min", 20],
                "ncovhist": ["min", 3000],
                "chinr": [-1, 15],
                "sharpnr": [-1, 1.5],
                "non_detections": ["min", 2000],
            },
        }
        self.params.update(new_default_params)

    def execute(self, df):
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.class_keys = ["AGN", "SN", "VS", "asteroid", "bogus"]
        pred, oids = self.predict_proba_service(df)
        pred_df = pd.DataFrame(index=oids, data=pred, columns=self.class_keys)

        return pred_df

    def predict_proba(self, data_array, features):
        raise Exception("Method predict_proba shouldn't be used!")

    def _predict_template(self, data_array, feature, variable):
        raise Exception("Method _predict_template shouldn't be used!")

    def predict_proba_service(self, df: pd.DataFrame):
        return self._predict_template_service(df, self.output_probabilities)

    def _predict_template_service(self, df: pd.DataFrame, variable):
        data_dict = self.df_to_dict(df)
        preprocessed_dataset = self.dataset_preprocessor.preprocess_dataset(
            Dataset(data_dict["images"], None, None, data_dict["metadata"])
        )
        data_array = preprocessed_dataset.data_array
        meta_data = preprocessed_dataset.meta_data
        predictions_dict = self.get_variables_by_batch(
            variables=[variable], data_array=data_array, features=meta_data
        )
        predictions_by_batch_dict = predictions_dict[variable]
        predictions = np.concatenate(
            predictions_by_batch_dict[general_keys.VALUES_PER_SAMPLE_IN_A_BATCH]
        )
        return predictions, data_dict["oid"]

    def df_to_dict(self, df: pd.DataFrame):
        df = self._feat_to_num(df)
        df = self.compute_extra_features(
            [
                self.ecliptic_coordinates,
                self.galactic_coordinates,
                self.approximate_non_detections,
            ],
            df,
        )
        n_samples = len(df)
        images = self.extract_images(df, self.stamp_keys)
        valid_images = images["shape"] == (63, 63, 3)
        metadata = self.extract_metadata(
            df, self.params[param_keys.FEATURES_NAMES_LIST]
        )

        images = images.image[valid_images]
        metadata = metadata[valid_images]
        data_dict = dict()
        data_dict["images"] = images.tolist()
        data_dict["metadata"] = metadata.values
        data_dict["oid"] = metadata.index
        return data_dict

    def extract_metadata(self, df, input_keys):
        metadata = df[input_keys]
        metadata.index = df["oid"]
        return metadata

    def stack_images(self, row, input_keys):
        return np.stack([row[k] for k in input_keys], axis=2)

    def extract_images(self, df, input_keys):
        oid = df["oid"]
        converted_images = pd.DataFrame(columns=input_keys)
        for key in input_keys:
            converted_images[key] = df[key].apply(get_image_from_bytes_stamp)

        stacked_images = pd.DataFrame()
        stacked_images["image"] = converted_images.apply(
            self.stack_images, axis=1, input_keys=self.stamp_keys
        )
        stacked_images["shape"] = stacked_images.image.apply(np.shape)
        stacked_images.index = oid
        return stacked_images

    def _feat_to_num(self, df):
        df.loc[df.isdiffpos == "f", "isdiffpos"] = 1
        df.loc[df.isdiffpos == "t", "isdiffpos"] = 0
        return df

    def ecliptic_coordinates(self, df):
        ecl = df.apply(
            lambda row: ephem.Ecliptic(
                ephem.Equatorial(
                    "%s" % (row.ra / 15.0), "%s" % row.dec, epoch=ephem.J2000
                )
            ),
            axis=1,
        )
        df["ecl_lat"] = ecl.apply(lambda row: np.rad2deg(row.lat))
        df["ecl_long"] = ecl.apply(lambda row: np.rad2deg(row.long))
        return df

    def galactic_coordinates(self, df):
        gal = df.apply(
            lambda row: ephem.Galactic(
                ephem.Equatorial(
                    "%s" % (row.ra / 15.0), "%s" % row.dec, epoch=ephem.J2000
                )
            ),
            axis=1,
        )
        df["gal_lat"] = gal.apply(lambda row: np.rad2deg(row.lat))
        df["gal_long"] = gal.apply(lambda row: np.rad2deg(row.long))
        return df

    def approximate_non_detections(self, df):
        df["non_detections"] = df["ncovhist"].values - df["ndethist"].values
        return df

    def compute_extra_features(self, function_list, df):
        for feature_function in function_list:
            df = feature_function(df)
        return df


if __name__ == "__main__":
    clf = StampClassifier()
    input_data = pd.read_pickle("../../predictions_in_ztf/one_stamp_20190624.pkl")
    pred = clf.execute(input_data)
    print(pred)
    print(pred.shape)
    pickle.dump(
        pred,
        open("../../predictions_in_ztf/one_stamp_bogus_20190626.pkl", "wb"),
        protocol=2,
    )
