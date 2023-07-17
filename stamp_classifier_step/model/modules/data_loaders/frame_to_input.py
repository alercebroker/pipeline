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
from modules.data_loaders.ztf_stamps_loader import ZTFLoader
from tqdm import tqdm
import pickle


def get_image_from_bytes_stamp(stamp_byte):
    with gzip.open(io.BytesIO(stamp_byte), "rb") as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data
    return img


class FrameToInput(ZTFLoader):
    def __init__(self, params):
        super().__init__(params)
        self.data_path = params[param_keys.DATA_PATH_TRAIN]
        self.converted_data_path = params[param_keys.CONVERTED_DATA_SAVEPATH]

    def _init_df_attributes(self, df):
        # self.data_frame = df
        self.n_points = len(df)
        self.class_names = np.unique(df["class"])
        self.class_dict = dict(
            zip(self.class_names, list(range(len(self.class_names))))
        )
        print(self.class_dict)
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.labels = []
        self.images = []
        self.metadata = []

    def get_datadict(self):
        loaded_data = pd.read_pickle(self.data_path)
        if not isinstance(loaded_data, pd.DataFrame):
            return loaded_data
            print("Recovering converted input")
        else:
            df = loaded_data
            self._init_df_attributes(loaded_data)
            for i in tqdm(range(self.n_points)):
                serie = df.loc[i]
                label = self.class_dict[serie["class"]]
                image_array = []
                for key in self.stamp_keys:
                    image_array.append(get_image_from_bytes_stamp(serie[key]))
                image_tensor = np.stack(image_array, axis=2)
                self.labels.append(label)
                self.images.append(image_tensor)

            aux_dict = {"labels": self.labels, "images": self.images, "features": None}
            pickle.dump(aux_dict, open(self.converted_data_path, "wb"), protocol=2)
            return aux_dict


if __name__ == "__main__":
    from parameters import general_keys

    DATA_PATH = os.path.join(PROJECT_PATH, "..", "pickles/training_set_Apr-27-2020.pkl")
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
        }
    )
    frame_to_input = FrameToInput(params)
    test_dict = frame_to_input.get_datadict()
    print(test_dict["images"][0].shape)
    print(np.unique(test_dict["labels"], return_counts=True))
    print(frame_to_input.class_dict)
