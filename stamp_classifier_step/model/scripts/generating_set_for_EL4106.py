# %%
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from models.visualizers.deepHits_lrp_visualizer import DeepHiTSLRPVisualizer
from parameters import param_keys, general_keys
import numpy as np
import pickle

data_path = os.path.join(
    PROJECT_PATH, "../pickles", "stamp_clf_training_set_Sep-13-2019.pkl"
)
# data_path = os.path.join(PROJECT_PATH, "/../pickles/converted_data.pkl")
n_classes = 5
params = {
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.WAIT_FIRST_EPOCH: False,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
    param_keys.TRAIN_HORIZON_INCREMENT: 10000,
    param_keys.TEST_SIZE: n_classes * 50,
    param_keys.VAL_SIZE: n_classes * 50,
    param_keys.NANS_TO: 0,
    param_keys.NUMBER_OF_CLASSES: n_classes,
    param_keys.CROP_SIZE: 21,
    param_keys.INPUT_IMAGE_SIZE: 21,
    param_keys.VALIDATION_MONITOR: general_keys.LOSS,
    param_keys.VALIDATION_MODE: general_keys.MIN,
    param_keys.ENTROPY_REG_BETA: 1,
}
model = DeepHiTSEntropyRegModel(params)
checkpoint_path = os.path.join(
    PROJECT_PATH,
    "results/entropy_reg/DeepHitsEntropyRegBeta1.0000_0_20190808-121100/checkpoints",
    "model",
)
model.load_model(checkpoint_path)
lrp_visualizer = DeepHiTSLRPVisualizer(model)

train_set, val_set, test_set = model._prepare_input(
    X=np.empty([]), y=np.empty([]), validation_data=None, test_data=None
)

print("Train %s" % str(np.unique(train_set.data_label, return_counts=True)))
print("Val %s" % str(np.unique(val_set.data_label, return_counts=True)))
print("Test %s" % str(np.unique(test_set.data_label, return_counts=True)))

dataset_dict = {
    "Train": {"data": train_set.data_array, "labels": train_set.data_label},
    "Validation": {"data": val_set.data_array, "labels": val_set.data_label},
    "Test": {"data": test_set.data_array, "labels": test_set.data_label},
}

save_path = os.path.join(PROJECT_PATH, "..", "pickles", "el4106_ztf_stamp_clf2.pkl")
pickle.dump(dataset_dict, open(save_path, "wb"), protocol=2)
