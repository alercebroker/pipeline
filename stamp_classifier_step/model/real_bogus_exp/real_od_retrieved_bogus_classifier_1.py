import os
import sys

"""
Comparing with Ashish boguses
"""

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_real_bog_nans_norm_crop_stamp_model import (
    DeepHiTSRealBogNanNormCropStampModel,
)
from parameters import param_keys, general_keys
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np
import pandas as pd


def get_df_dataset_from_name(
    model: DeepHiTSRealBogNanNormCropStampModel, path: str
) -> Dataset:
    params_copy = model.params.copy()
    params_copy.update({param_keys.DATA_PATH_TRAIN: path})
    frame_to_input = FrameToInput(params_copy)
    frame_to_input.dataset_preprocessor.set_pipeline(
        model.dataset_preprocessor.preprocessing_pipeline[:-1]
    )
    return frame_to_input.get_preprocessed_dataset_unsplitted()


def recall_over_specific_class(positive_class_value, predictions, labels):
    Positives = np.sum([labels == positive_class_value])
    TP = np.sum((predictions == labels)[labels == positive_class_value])
    # print(Positives)
    # print(TP)
    # print(TP / Positives)
    return TP / Positives


if __name__ == "__main__":
    load_GeoT_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/OD_retieved_1"
    data_name = "converted_pancho_septiembre.pkl"
    data_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data"
    data_path = os.path.join(data_folder, data_name)

    n_classes = 2
    params = {
        param_keys.RESULTS_FOLDER_NAME: "real_bogus",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 390,
        param_keys.VAL_SIZE: n_classes * 195,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: None,
        # param_keys.VALIDATION_RANDOM_SEED: 42,
    }
    model = DeepHiTSRealBogNanNormCropStampModel(
        params, model_name="Real_Bog_Retrieve_1"
    )

    # dataset_loading
    mayor_Test_set_8_dataset = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Mayor_Test_set_(8)_dataset.pkl")
    )
    # Train set (3)U(4)U(7)
    inliers_Val_3 = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Inliers_Val_(3).pkl")
    )
    print("Inliers_Val_(3)", inliers_Val_3.shape)
    inliers_Train_4 = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Inliers_Train_(4).pkl")
    )
    print("Inliers_Train_(4)", inliers_Train_4.shape)
    train_geoTransform_3_U_4 = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Train_Geotransform_(3)_U_(4).pkl")
    )
    print("Train_Geotransform_(3)_U_(4)", train_geoTransform_3_U_4.shape)
    detected_Boguses_7 = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Detected_Boguses_(7).pkl")
    )
    # retrieve_folder = '/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/OD_retieved'
    # detected_Boguses_7 = pd.read_pickle(os.path.join(retrieve_folder, 'new_od_boguses.pkl'))
    print("Detected_Boguses_(7)", detected_Boguses_7.shape)
    train_set_inliers = np.concatenate([inliers_Val_3, inliers_Train_4])
    x_train = np.concatenate([train_set_inliers, detected_Boguses_7])
    y_train = np.concatenate(
        [np.ones(len(train_set_inliers)), np.zeros(len(detected_Boguses_7))]
    )
    # Test set (5) U (6)
    inliers_Test_5 = pd.read_pickle(
        os.path.join(load_GeoT_folder, "Inliers_Test_(5).pkl")
    )
    print("Inliers_Test_(5)", inliers_Test_5.shape)
    all_bogus_6 = pd.read_pickle(os.path.join(load_GeoT_folder, "ALL_Bogus_(6).pkl"))
    print("ALL_Boguses_(6)", all_bogus_6.shape)
    x_test_5_U_6 = np.concatenate([inliers_Test_5, all_bogus_6])
    y_test_5_U_6 = np.concatenate(
        [np.ones(len(inliers_Test_5)), np.zeros(len(all_bogus_6))]
    )
    # TNS (9)
    tns_name = "tns_confirmed_sn.pkl"
    tns_path = os.path.join(data_folder, "converted_" + tns_name)
    tns_dataset = get_df_dataset_from_name(model, tns_path)
    print("tns.shape: ", tns_dataset.data_array.shape)
    # ALerce Bogus (1)
    alerce_bogus_data_name = "bogus_juliano_franz_pancho.pkl"
    alerce_bogus_path = os.path.join(data_folder, "converted_" + alerce_bogus_data_name)
    alerce_bogus_dataset = get_df_dataset_from_name(model, alerce_bogus_path)
    print("ALERCE_bogus.shape: ", alerce_bogus_dataset.data_array.shape)

    # trainer = Trainer(params)
    # train_set, val_set, test_set = model._data_init()
    # print(test_set.data_array[12,2,3,2])
    # print(Mayor_Test_set_8_dataset.data_array[12, 2, 3, 2])

    model.fit(
        x_train,
        y_train,
        test_data=(
            mayor_Test_set_8_dataset.data_array,
            mayor_Test_set_8_dataset.data_label,
        ),
    )
    model.evaluate(x_test_5_U_6, y_test_5_U_6, set="Test_(5)_U_(6)")

    alerce_bogus_predictions = model.predict(alerce_bogus_dataset.data_array)
    alerce_bogus_recall = recall_over_specific_class(
        0, alerce_bogus_predictions, np.zeros(len(alerce_bogus_predictions))
    )
    print("Alerce_bogus_(1) recall: ", alerce_bogus_recall)

    all_bogus_predictions = model.predict(all_bogus_6)
    all_bogus_recall = recall_over_specific_class(
        0, all_bogus_predictions, np.zeros(len(all_bogus_predictions))
    )
    print("All_bogus_(6) recall: ", all_bogus_recall)

    tns_predictions = model.predict(tns_dataset.data_array)
    tns_recall = recall_over_specific_class(
        1, tns_predictions, np.ones(len(tns_predictions))
    )
    print("tns_(9) recall: ", tns_recall)

    inliers_test_predictions = model.predict(inliers_Test_5)
    inliers_recall = recall_over_specific_class(
        1, inliers_test_predictions, np.ones(len(inliers_test_predictions))
    )
    print("inliers_test_(5) recall: ", inliers_recall)

"""
Total training time: 0:03:31

Evaluating final model...
Best model @ it 10430.
Validation loss 0.14380

Validation Metrics: loss                       0.143801, accuracy 0.948718
Normalized confusion matrix Acc 0.9487
[[0.92307692 0.07692308]
 [0.02564103 0.97435897]]
Confusion matrix, without normalization Acc 0.9487
[[180  15]
 [  5 190]]
Test Metrics: loss                             0.107389, accuracy 0.975641
Normalized confusion matrix Acc 0.9756
[[0.97692308 0.02307692]
 [0.02564103 0.97435897]]
Confusion matrix, without normalization Acc 0.9756
[[381   9]
 [ 10 380]]
Test_(5)_U_(6) Metrics: loss                   0.170771, accuracy 0.948286
Normalized confusion matrix Acc 0.9483
[[0.93936321 0.06063679]
 [0.04279051 0.95720949]]
Confusion matrix, without normalization Acc 0.9483
[[4632  299]
 [ 211 4720]]
Alerce_bogus_(1) recall:  0.9275964391691395
All_bogus_(6) recall:  0.9393632123301562
tns_(9) recall:  0.8582995951417004
inliers_test_(5) recall:  0.9572094909754614

"""
