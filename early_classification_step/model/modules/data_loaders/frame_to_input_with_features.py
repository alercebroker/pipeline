import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from parameters import param_keys
import numpy as np
import pandas as pd
import gzip
from astropy.io import fits
import io
from tqdm import tqdm
from modules.data_loaders.frame_to_input import FrameToInput, get_image_from_bytes_stamp
import pickle
from parameters.errors import check_data_frama_contain_features
import multiprocessing
from joblib import Parallel, delayed
from typing import Dict
from modules.data_set_generic import Dataset
from parameters import general_keys


def get_image_from_bytes_stamp(stamp_byte):
    with gzip.open(io.BytesIO(stamp_byte), "rb") as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data
    return img


def _subprocess_by_serie(serie, class_dict, stamp_keys, features_keys):
    label = class_dict[serie["class"]]
    image_array = []
    for key in stamp_keys:
        image_array.append(get_image_from_bytes_stamp(serie[key]))
    image_tensor = np.stack(image_array, axis=2)
    feature = serie[features_keys].values
    aux_dict = {
        general_keys.LABELS: label,
        general_keys.IMAGES: image_tensor,
        general_keys.FEATURES: feature,
    }
    return aux_dict


class FrameToInputWithFeatures(FrameToInput):
    def __init__(self, params):
        self.params = self.get_default_params()
        self.params.update(params)
        super().__init__(self.params)
        self.features_names_list = self.params[param_keys.FEATURES_NAMES_LIST]
        self.aux_converted_path = self.params[param_keys.CONVERTED_DATA_SAVEPATH]

    def get_default_params(self):
        params = {
            param_keys.FEATURES_NAMES_LIST: [],
            # param_keys.BATCH_SIZE: None,
            # param_keys.TEST_SIZE: 0,
            # param_keys.TEST_RANDOM_SEED: 0,
            # param_keys.VAL_SIZE: 0,
            # param_keys.VALIDATION_RANDOM_SEED: 0,
            # param_keys.CHANNELS_TO_USE: [0, 1, 2],
            # param_keys.NANS_TO: 0,
            # param_keys.CROP_SIZE: 21,
            # param_keys.CONVERTED_DATA_SAVEPATH: os.path.join(
            #     PROJECT_PATH, "..", "pickles", "converted_data.pkl")
        }
        return params

    def _init_df_attributes(self, df: pd.DataFrame):
        # self.data_frame = df
        self.n_cpus = multiprocessing.cpu_count()
        check_data_frama_contain_features(df, self.features_names_list)
        self.n_points = len(df)
        self.class_names = np.unique(df["class"])
        self.class_dict = dict(
            zip(self.class_names, list(range(len(self.class_names))))
        )
        print(self.class_dict)
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.labels = []
        self.images = []
        self.features = []

    def _feat_to_num(self, df):
        df.loc[df.isdiffpos == "f", "isdiffpos"] = 1
        df.loc[df.isdiffpos == "t", "isdiffpos"] = 0
        return df

    def _group_multiproc_dicts(self, multiproc_result_dicts):
        bar = {
            k: [d.get(k) for d in multiproc_result_dicts]
            for k in set().union(*multiproc_result_dicts)
        }
        return bar

    def get_datadict(self) -> Dict[str, list]:
        loaded_data = pd.read_pickle(self.data_path)
        if not isinstance(loaded_data, pd.DataFrame):
            print("Recovering converted input")
            return loaded_data

        else:
            df = loaded_data
            self._init_df_attributes(df)
            object_ids = df["oid"].tolist()
            if "isdiffpos" in df.columns:
                df = self._feat_to_num(df)

            results = Parallel(n_jobs=self.n_cpus)(
                delayed(_subprocess_by_serie)(
                    df.loc[i],
                    self.class_dict,
                    self.stamp_keys,
                    self.features_names_list,
                )
                for i in tqdm(range(self.n_points))
            )

            del df
            results_dict = self._group_multiproc_dicts(results)
            del results
            aux_dict = {
                general_keys.LABELS: results_dict[general_keys.LABELS],
                general_keys.IMAGES: results_dict[general_keys.IMAGES],
                general_keys.FEATURES: results_dict[general_keys.FEATURES],
                general_keys.OBJECT_IDS: object_ids,
            }
            del results_dict
            if self.params[param_keys.CONVERTED_DATA_SAVEPATH] is not None:
                print(
                    "Dumping to %s..." % self.params[param_keys.CONVERTED_DATA_SAVEPATH]
                )
                pickle.dump(aux_dict, open(self.converted_data_path, "wb"), protocol=2)
            return aux_dict

    def set_dumping_data_to_pickle(self, dump_to_pickle=True):
        if dump_to_pickle:
            self.params[param_keys.CONVERTED_DATA_SAVEPATH] = self.aux_converted_path
        else:
            self.params[param_keys.CONVERTED_DATA_SAVEPATH] = None

    def get_preprocessed_datasets_splitted(self) -> Dict[str, Dataset]:
        self.dataset_preprocessor.set_use_of_feature_normalizer(
            use_feature_normalizer=False
        )
        self.dataset_preprocessor.verbose_warnings = False
        dataset = self.get_preprocessed_dataset_unsplitted()
        datasets_dict = self._init_splits_dict()
        self.data_splitter.set_dataset_obj(dataset)
        (
            train_dataset,
            test_dataset,
            val_dataset,
        ) = self.data_splitter.get_train_test_val_set_objs()
        self.dataset_preprocessor.set_use_of_feature_normalizer(
            use_feature_normalizer=True
        )
        self.dataset_preprocessor.verbose = False
        datasets_dict[
            general_keys.TRAIN
        ] = self.dataset_preprocessor.preprocess_dataset(train_dataset)
        datasets_dict[general_keys.TEST] = self.dataset_preprocessor.preprocess_dataset(
            test_dataset
        )
        datasets_dict[
            general_keys.VALIDATION
        ] = self.dataset_preprocessor.preprocess_dataset(val_dataset)
        self.dataset_preprocessor.verbose = True
        self.dataset_preprocessor.verbose_warnings = True
        return datasets_dict


if __name__ == "__main__":

    DATA_PATH = os.path.join(
        PROJECT_PATH, "..", "pickles/full_unlabeled_set/combined_set/full_frame.pkl"
    )
    # 'pickles/converted_data.pkl')
    N_CLASSES = 5
    params = param_keys.default_params.copy()
    params.update(
        {
            param_keys.RESULTS_FOLDER_NAME: "gal_ecl_coord_beta_search",
            param_keys.DATA_PATH_TRAIN: DATA_PATH,
            param_keys.WAIT_FIRST_EPOCH: False,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
            param_keys.TRAIN_HORIZON_INCREMENT: 10000,
            param_keys.TEST_SIZE: N_CLASSES * 100,
            param_keys.VAL_SIZE: N_CLASSES * 100,
            param_keys.NANS_TO: 0,
            param_keys.NUMBER_OF_CLASSES: N_CLASSES,
            param_keys.CROP_SIZE: 21,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.VALIDATION_MONITOR: general_keys.LOSS,
            param_keys.VALIDATION_MODE: general_keys.MIN,
            param_keys.ENTROPY_REG_BETA: 0.5,
            param_keys.LEARNING_RATE: 1e-4,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 32,
            param_keys.KERNEL_SIZE: 3,
            param_keys.FEATURES_NAMES_LIST: [
                "non_detections",
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
                "oid",
            ],
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
    )
    frame_to_input = FrameToInputWithFeatures(params)
    frame_to_input.dataset_preprocessor.set_pipeline(
        [
            frame_to_input.dataset_preprocessor.image_check_single_image,
            frame_to_input.dataset_preprocessor.image_clean_misshaped,
            frame_to_input.dataset_preprocessor.image_select_channels,
            frame_to_input.dataset_preprocessor.image_crop_at_center,
            frame_to_input.dataset_preprocessor.image_normalize_by_image,
            frame_to_input.dataset_preprocessor.image_nan_to_num,
            frame_to_input.dataset_preprocessor.features_clip,
            frame_to_input.dataset_preprocessor.features_normalize,
        ]
    )
    frame_to_input.set_dumping_data_to_pickle(dump_to_pickle=False)

    # # test dict methods
    # test_dict = frame_to_input.get_datadict()
    # print(np.unique(test_dict[general_keys.LABELS], return_counts=True))
    # print(test_dict[general_keys.IMAGES][0].shape)
    # print(np.array(test_dict[general_keys.FEATURES])[:, -1][-5:])
    # print(np.array(test_dict[general_keys.FEATURES])[:,
    #       -1].tolist() == test_dict[general_keys.OBJECT_IDS])

    frame_to_input_2 = FrameToInputWithFeatures(params)
    frame_to_input_2.dataset_preprocessor.set_pipeline(
        [
            frame_to_input_2.dataset_preprocessor.image_check_single_image,
            frame_to_input_2.dataset_preprocessor.image_clean_misshaped,
            frame_to_input_2.dataset_preprocessor.image_select_channels,
        ]
    )
    frame_to_input_2.set_dumping_data_to_pickle(dump_to_pickle=False)

    print("\n test preprocessed features and non preprocessed equal oid")
    preprocessed_splits_1 = frame_to_input.get_preprocessed_datasets_splitted()
    preprocessed_splits_2 = frame_to_input_2.get_preprocessed_datasets_splitted()
    split_name = general_keys.TRAIN
    print(
        np.mean(
            (
                preprocessed_splits_1[split_name].meta_data[:, -1]
                == preprocessed_splits_1[split_name].meta_data[:, -1]
            )
        )
    )
    print(np.mean(preprocessed_splits_1[split_name].meta_data[:, 0]))
    print(np.mean(preprocessed_splits_2[split_name].meta_data[:, 0]))
    split_name = general_keys.TEST
    print(
        np.mean(
            (
                preprocessed_splits_1[split_name].meta_data[:, -1]
                == preprocessed_splits_1[split_name].meta_data[:, -1]
            )
        )
    )
    print(np.mean(preprocessed_splits_1[split_name].meta_data[:, 0]))
    print(np.mean(preprocessed_splits_2[split_name].meta_data[:, 0]))
    split_name = general_keys.VALIDATION
    print(
        np.mean(
            (
                preprocessed_splits_1[split_name].meta_data[:, -1]
                == preprocessed_splits_1[split_name].meta_data[:, -1]
            )
        )
    )
    print(np.mean(preprocessed_splits_1[split_name].meta_data[:, 0]))
    print(np.mean(preprocessed_splits_2[split_name].meta_data[:, 0]))

    # # test clipped features
    # preprocessed_dataset = frame_to_input.get_preprocessed_dataset_unsplitted()
    # print(preprocessed_dataset.meta_data[-5:, -1])
    # print(np.max(preprocessed_dataset.meta_data[:, 0]))
    # print(np.min(preprocessed_dataset.meta_data[:, 0]))
    # print(np.max(np.array(test_dict[general_keys.FEATURES])[:, 0]))
    # print(np.min(np.array(test_dict[general_keys.FEATURES])[:, 0]))

    # # test preprocessed features
    # print('\n test preprocessed features')
    # preprocessed_splits = frame_to_input.get_preprocessed_datasets_splitted()
    # print(np.mean(np.array(test_dict[general_keys.FEATURES])[:, 0]))
    # print(np.std(np.array(test_dict[general_keys.FEATURES])[:, 0]))
    # split_name = general_keys.TRAIN
    # print(preprocessed_splits[split_name].meta_data[-5:, -1])
    # print(np.mean(preprocessed_splits[split_name].meta_data[:, 0]))
    # print(np.std(preprocessed_splits[split_name].meta_data[:, 0]))
    # split_name = general_keys.TEST
    # print(preprocessed_splits[split_name].meta_data[-5:, -1])
    # print(np.mean(preprocessed_splits[split_name].meta_data[:, 0]))
    # print(np.std(preprocessed_splits[split_name].meta_data[:, 0]))
    # split_name = general_keys.VALIDATION
    # print(preprocessed_splits[split_name].meta_data[-5:, -1])
    # print(np.mean(preprocessed_splits[split_name].meta_data[:, 0]))
    # print(np.std(preprocessed_splits[split_name].meta_data[:, 0]))

    # # test train oids
    # raw_splits = frame_to_input.get_raw_datasets_splitted()
    # preprocessed_splits = frame_to_input.get_preprocessed_datasets_splitted()
    # oids_split = frame_to_input.get_object_ids_splitted()
    #
    # split_name = general_keys.TRAIN
    # print(split_name)
    # print(len(np.array(raw_splits[split_name].meta_data)[:,
    #       -1].tolist()))
    # print(len(np.array(preprocessed_splits[split_name].meta_data)[:,
    #       -1].tolist()))
    # print(len(oids_split[split_name][:]))
    # print(np.array(raw_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
    # print(np.array(preprocessed_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
    #
    # split_name = general_keys.TEST
    # print(split_name)
    # print(np.array(raw_splits[split_name].meta_data)[-5:,
    #       -1].tolist())
    # print(np.array(preprocessed_splits[split_name].meta_data)[-5:,
    #       -1].tolist())
    # print(oids_split[split_name][-5:])
    # print(np.array(raw_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
    # print(np.array(preprocessed_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
    #
    # split_name = general_keys.VALIDATION
    # print(split_name)
    # print(np.array(raw_splits[split_name].meta_data)[-5:,
    #       -1].tolist())
    # print(np.array(preprocessed_splits[split_name].meta_data)[-5:,
    #       -1].tolist())
    # print(oids_split[split_name][-5:])
    # print(np.array(raw_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
    # print(np.array(preprocessed_splits[split_name].meta_data)[:,
    #       -1].tolist() == oids_split[split_name])
