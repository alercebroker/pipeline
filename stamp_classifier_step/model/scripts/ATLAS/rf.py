import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_atlas import DeepHiTSAtlasWithFeatures
import numpy as np
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from scripts.ATLAS.pvalues_to_leave_out_metadata import (
    get_feature_idxs_without_p,
    get_feature_names,
)


from modules.confusion_matrix import plot_confusion_matrix

class_names = np.array(["cr", "streak", "burn", "scar", "kast", "spike", "noise"])


if __name__ == "__main__":
    N_TRAIN = 1
    data_path = os.path.join(
        PROJECT_PATH, "..", "ATLAS", "atlas_data_with_metadata.pkl"
    )
    n_classes = 7
    fetures_indices_used = list(range(73))
    print(fetures_indices_used)
    print(len(fetures_indices_used))
    params = {
        param_keys.BATCH_SIZE: 32,
        param_keys.RESULTS_FOLDER_NAME: "atlas_with_features_all_metadata",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 1000,
        param_keys.TRAIN_HORIZON_INCREMENT: 1000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: None,
        param_keys.INPUT_IMAGE_SIZE: 101,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.ITERATIONS_TO_VALIDATE: 10,
        param_keys.FEATURES_NAMES_LIST: fetures_indices_used,
    }

    params.update(
        {
            param_keys.CROP_SIZE: 63,
            param_keys.INPUT_IMAGE_SIZE: 63,
        }
    )
    data_preprocessor = ATLASDataPreprocessor(params)
    pipeline = [
        data_preprocessor.image_check_single_image,
        data_preprocessor.images_to_gray_scale,
        data_preprocessor.image_crop_at_center,
        data_preprocessor.image_normalize_by_image_1_1,
        data_preprocessor.image_nan_to_num,
        data_preprocessor.features_normalize,
    ]
    data_preprocessor.set_pipeline(pipeline)
    params.update(
        {
            param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor,
            param_keys.N_INPUT_CHANNELS: 1,
        }
    )

    aux_model = DeepHiTSAtlasWithFeatures(params)
    train_set, val_set, test_set = aux_model._prepare_input()
    print(train_set.meta_data.shape)
    print(val_set.meta_data.shape)
    print(test_set.meta_data.shape)
    print(np.unique(train_set.data_label, return_counts=True))
    print(np.mean(train_set.meta_data, axis=0).mean())
    print(np.std(train_set.meta_data, axis=0).mean())
    print(np.mean(test_set.meta_data, axis=0).mean())
    print(np.std(test_set.meta_data, axis=0).mean())

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=0, n_jobs=-1
    )
    clf.fit(train_set.meta_data, train_set.data_label)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    len_nonzero = np.sum(importances >= 0.01)

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len_nonzero):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    y_pred_test = clf.predict(test_set.meta_data)
    print("test acc: ", np.mean(y_pred_test == test_set.data_label))
    plot_confusion_matrix(test_set.data_label, y_pred_test, show=True)
    y_pred_val = clf.predict(val_set.meta_data)
    print("val acc: ", np.mean(y_pred_val == val_set.data_label))

    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    palette = sns.color_palette("bright", 7)

    feature_names = get_feature_names()
    feature_names_used = []
    for idx_used in fetures_indices_used:
        feature_names_used.append(feature_names[idx_used])
    print(feature_names_used)

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("%i most important features" % len_nonzero)
    print(importances)
    print(indices)
    print(len(indices))
    plt.bar(
        range(len_nonzero),
        importances[indices][:len_nonzero],
        color="r",
        # yerr=std[indices][:len_nonzero],
        align="center",
        label="Feature Importance",
    )
    # erbpl1 = plt.errorbar(range(len_nonzero), importances[indices][:len_nonzero],
    #                       yerr=std[indices][:len_nonzero], fmt='o', label='Std of 100 estimators')
    plt.xticks(
        range(len_nonzero),
        np.array(feature_names_used)[indices][:len_nonzero],
        rotation=90,
    )
    # plt.xticks(range(len_nonzero),
    #            indices[:len_nonzero], rotation=90)
    plt.xlim([-1, len_nonzero])
    plt.legend()
    plt.show()

    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30

    tsne = TSNE()
    X_embedded = tsne.fit_transform(train_set.meta_data)

    train_labels = []
    for label_i in train_set.data_label:
        train_labels.append(class_names[label_i])
    lm = sns.scatterplot(
        X_embedded[:, 0],
        X_embedded[:, 1],
        hue=train_labels,
        legend="full",
        palette=palette,
    ).set_title("TSNE over Train Data", fontsize=12)
    plt.show()
