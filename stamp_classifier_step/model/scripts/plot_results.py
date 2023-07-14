import os
import sys
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from scripts.plot_confusion_matrix import plot_confusion_matrix
from parameters import param_keys

# data_path = os.path.join("../../pickles", 'corrected_oids_alerts.pkl')
# "all_alerts_training.pkl")
data_path = "../../pickles/converted_data.pkl"

n_classes = 5
params = {
    param_keys.RESULTS_FOLDER_NAME: "crop_at_center",
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.WAIT_FIRST_EPOCH: False,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
    param_keys.TRAIN_HORIZON_INCREMENT: 10000,
    param_keys.TEST_SIZE: n_classes * 50,
    param_keys.VAL_SIZE: n_classes * 50,
    param_keys.NANS_TO: 0,
    param_keys.NUMBER_OF_CLASSES: n_classes,
    param_keys.CROP_SIZE: 21,
    param_keys.INPUT_IMAGE_SIZE: 21,
}

model = DeepHiTSNanNormCropStampModel(params)
dataset = model._data_init()
checkpoint_path = os.path.join("../results/best_model_so_far/checkpoints", "model")
model.load_model(checkpoint_path)
model.evaluate(dataset[1].data_array, dataset[1].data_label, set="Val")
model.evaluate(dataset[2].data_array, dataset[2].data_label, set="Test")

class_names = np.array(["AGN", "SN", "VS", "asteroid", "bogus"])
pred = model.predict(dataset[0].data_array)
plot_confusion_matrix(
    dataset[0].data_label,
    pred,
    classes=class_names,
    normalize=True,
    save_path="../results/best_model_so_far/cm.png",
)
