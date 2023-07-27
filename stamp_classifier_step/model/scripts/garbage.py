# %%

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import os, sys

PROJECT_PATH = os.path.join("..")
sys.path.append(PROJECT_PATH)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import datashader as ds
# from datashader import transfer_functions
# import bokeh.plotting as bp
# from matplotlib.cm import viridis
# import datashader as ds
# import holoviews as hv

# print(hv.__version__)
# from holoviews import opts
# from holoviews.operation.datashader import datashade, shade, spread, dynspread, rasterize
# import holoviews.plotting.bokeh

# hv.extension('bokeh', 'matplotlib')

PROJECT_PATH = os.path.join("..")
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures
import pickle
from notebooks.evaluate_unlabeled_data import get_predictions_of_chunk
from modules.data_set_generic import Dataset
from scripts.plot_confusion_matrix import plot_confusion_matrix, plot_cm_std
from sklearn.metrics import confusion_matrix
from glob import glob
import tensorflow as tf

# import ephem

# %% md

# Confusion matrix

# %%

# instance model and load weights
data_dir = "../../pickles/"
DATA_PATH = os.path.join(os.path.join(data_dir, "training_set_May-06-2020.pkl"))
# DATA_PATH = "../../pickles/converted_data.pkl"
N_CLASSES = 5

features_keys = [
    "oid",
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
]

n_classes = 5
params = {
    param_keys.RESULTS_FOLDER_NAME: "aux_model",
    param_keys.DATA_PATH_TRAIN: DATA_PATH,
    param_keys.WAIT_FIRST_EPOCH: False,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
    param_keys.TRAIN_HORIZON_INCREMENT: 10000,
    param_keys.TEST_SIZE: n_classes * 100,
    param_keys.VAL_SIZE: n_classes * 100,
    param_keys.NANS_TO: 0,
    param_keys.NUMBER_OF_CLASSES: n_classes,
    param_keys.CROP_SIZE: 21,
    param_keys.INPUT_IMAGE_SIZE: 21,
    param_keys.LEARNING_RATE: 0.0001,
    param_keys.VALIDATION_MONITOR: general_keys.LOSS,
    param_keys.VALIDATION_MODE: general_keys.MIN,
    param_keys.ENTROPY_REG_BETA: 0.8,
    param_keys.DROP_RATE: 0.8,
    param_keys.BATCH_SIZE: 32,
    param_keys.KERNEL_SIZE: 3,
    param_keys.FEATURES_NAMES_LIST: [
        "oid",
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
aux_model = DeepHiTSWithFeaturesEntropyReg(params)
params = aux_model.params
train_set, val_set, test_set = aux_model._prepare_input()
aux_model.close()
del aux_model
# aux_model.close()
# del aux_model

# frame_to_input = FrameToInputWithFeatures(params)
# frame_to_input.set_dumping_data_to_pickle(False)
# frame_to_input.dataset_preprocessor.set_pipeline(
#          [frame_to_input.dataset_preprocessor.image_check_single_image,
#           frame_to_input.dataset_preprocessor.image_clean_misshaped,
#           frame_to_input.dataset_preprocessor.image_select_channels,
#           frame_to_input.dataset_preprocessor.image_crop_at_center,
#           frame_to_input.dataset_preprocessor.image_normalize_by_image,
#           frame_to_input.dataset_preprocessor.image_nan_to_num,
#           frame_to_input.dataset_preprocessor.features_clip,
#           frame_to_input.dataset_preprocessor.features_normalize
#           ])


# data_dict = frame_to_input.get_preprocessed_datasets_splitted()
# train_set, val_set, test_set = data_dict["train"], data_dict["validation"], data_dict["test"]

# %%

test_set

# %%

params.update(
    {
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
        param_keys.DATA_PATH_TRAIN: None,
    }
)

model_paths = glob(
    os.path.join(
        "../results/LA_MEJOR_WEA/",
        "DeepHits_EntropyRegBeta0.8000_batch32_lr0.00100_droprate0.8000_inputsize21_filtersize3_*",
    )
)
print(model_paths)

feature_model_list = []
for model_path in model_paths:
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    with graph.as_default():
        model = DeepHiTSWithFeaturesEntropyReg(params, session=sess)
        checkpoint_path_best_so_far = os.path.join(model_path, "checkpoints", "model")
        model.load_model(checkpoint_path_best_so_far)
        feature_model_list.append(model)

model_paths = glob(
    os.path.join(
        "../results/best_model_without_features_extra_layer/",
        "DeepHits_EntropyRegBeta0.8000_batch32_lr0.00100_droprate0.8000_inputsize21_filtersize3_*",
    )
)
model_list = []
for model_path in model_paths:
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    with graph.as_default():
        model = DeepHiTSEntropyRegModel(params, session=sess)
        checkpoint_path_best_so_far = os.path.join(model_path, "checkpoints", "model")
        model.load_model(checkpoint_path_best_so_far)
        model_list.append(model)

# %%

normalize_cm = True
class_names = np.array(["AGN", "SN", "VS", "asteroid", "bogus"])
all_cms = []

predictions_with_features_per_model = []

acc = []
for model in feature_model_list:
    y_pred = model.predict(test_set.data_array, features=test_set.meta_data[:, 1:])
    acc.append(np.mean(y_pred == test_set.data_label))
    predictions_with_features_per_model.append(
        {
            "prediction": y_pred,
            "prob": model.predict_proba(
                test_set.data_array, features=test_set.meta_data[:, 1:]
            ),
        }
    )
    cm = confusion_matrix(test_set.data_label, y_pred)
    if normalize_cm:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix Acc %.4f" % (np.trace(cm) / np.sum(cm)))
    all_cms.append(cm)

print(np.mean(acc), np.std(acc))
all_cms = np.stack(all_cms)
mean_cm, std_cm = np.mean(all_cms, axis=0), np.std(all_cms, axis=0)

# % matplotlib
# inline
p = plot_cm_std(
    mean_cm=mean_cm,
    std_cm=std_cm,
    title="",
    classes=class_names,
    normalize=True,
    label_fontsize=14,
    axis_fontsize=14,
    colorbar=False,
    savepath="revision_plots/mean_matrix.pdf",
)
plt.show()
