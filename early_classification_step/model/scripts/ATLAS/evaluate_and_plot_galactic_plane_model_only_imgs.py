import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_atlas import DeepHiTSAtlas
from modules.confusion_matrix import plot_confusion_matrix
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.data_set_generic import Dataset
from modules.coords import ec2gal
import ephem
from modules.utils import save_pickle


def plot_class_atlas_hist(
    blind_preds, class_names_short=np.array(["artifact", "kast", "streak"])
):
    label_values, label_counts = np.unique(blind_preds, return_counts=True)
    plt.bar(
        label_values,
        label_counts,
        align="center",
        label="Blind data distribution",
        log=True,
    )
    plt.xticks(label_values, class_names_short[label_values], rotation=90)
    # plt.xticks(range(len_nonzero),
    #            indices[:len_nonzero], rotation=90)
    plt.ylim([1, 10**5])
    plt.xlim([-1, len(label_values)])
    plt.ylabel("N° Samples")
    plt.xlabel("Label Names")
    plt.legend()
    plt.show()


def galactic_coordinates(ra_list, dec_list):
    gal_lat = []
    gal_long = []
    for i in range(len(ra_list)):
        # gal = ephem.Galactic(ephem.Equatorial('%s' % (np.rad2deg(ra_list[i]) / 15.),
        #                        '%s' % np.rad2deg(dec_list[i]), epoch=ephem.J2000))
        gal = ephem.Galactic(
            ephem.Equatorial(
                "%s" % (ra_list[i] / 15.0), "%s" % dec_list[i], epoch=ephem.J2000
            )
        )
        gal_lat.append(np.rad2deg(gal.lat))
        gal_long.append(np.rad2deg(gal.long))
    return np.array(gal_lat), np.array(gal_long)


def plot_galactic_plane_of_blinds(
    blind_dataset_preprop,
    blind_preds,
    class_names=np.array(["artifact", "kast", "streak"]),
):
    f, ax = plt.subplots(3, 3, figsize=(3 * 10, 3 * 6))
    plt.subplots_adjust(wspace=0.0, hspace=0.15)
    ax = ax.flatten()

    class_names_title = ["Predicted %s" % name for name in list(class_names)]
    titles = ["Unlabeled data"] + class_names_title
    cmin_values = [0.1] * len(titles)
    fontsize = 18
    dims = ["gal_long", "gal_lat"]

    ecliptic_lat = np.linspace(0, 0, num=500)
    ecliptic_longi = np.linspace(0, 360, num=500)
    suns_galact_long, suns_galact_lat = ec2gal(ecliptic_longi, ecliptic_lat)

    ras, decs = (
        blind_dataset_preprop.meta_data[:, 0],
        blind_dataset_preprop.meta_data[:, 1],
    )
    # print('')
    # data_aux = {'ra': ras, 'dec': decs}
    # save_pickle(data_aux, 'blind_ra_dec_dict.pkl')

    gal_lat, gal_long = galactic_coordinates(ras, decs)
    for i in range(len(titles)):
        if i == 0:
            _, _, _, im = ax[i].hist2d(
                gal_long,
                gal_lat,
                (300, 300),
                cmap="viridis",
                # vmax=vmax_values[i],
                cmin=cmin_values[i],
            )
        else:
            pred_i_idxs = np.argwhere(blind_preds == i - 1).flatten()
            # print(pred_i_idxs)
            gal_lat_i = gal_lat[pred_i_idxs]
            gal_long_i = gal_long[pred_i_idxs]
            _, _, _, im = ax[i].hist2d(
                gal_long_i, gal_lat_i, (300, 300), cmap="viridis", cmin=cmin_values[i]
            )
        ax[i].tick_params(axis="both", labelsize=fontsize - 2)
        cbar = plt.colorbar(im, ax=ax[i], pad=0)
        if i == 0 or i == 3:
            ax[i].set_ylabel("Galactic Latitude", fontsize=fontsize)
        if i == 3 or i == 4 or i == 5:
            ax[i].set_xlabel("Galactic Longitude", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 4)
        ax[i].set_title(titles[i], fontsize=fontsize)
        ax[i].plot(suns_galact_long, suns_galact_lat, "--", lw=3)
        ax[i].plot(suns_galact_long, suns_galact_lat, "-k", lw=6)
        ax[i].plot(suns_galact_long, suns_galact_lat, "-y", lw=4)
    plt.show()


if __name__ == "__main__":
    data_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "atlas_data.pkl")
    n_classes = 3
    params = {
        param_keys.BATCH_SIZE: 32,
        param_keys.RESULTS_FOLDER_NAME: "testing_atlas",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 1,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 5000,
        param_keys.TRAIN_HORIZON_INCREMENT: 1000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 63,
        param_keys.INPUT_IMAGE_SIZE: 63,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.ITERATIONS_TO_VALIDATE: 10,
    }
    data_preprocessor = ATLASDataPreprocessor(params)
    pipeline = [
        data_preprocessor.image_check_single_image,
        data_preprocessor.image_clean_misshaped,
        data_preprocessor.image_select_channels,
        data_preprocessor.images_to_gray_scale,
        data_preprocessor.image_crop_at_center,
        data_preprocessor.image_normalize_by_image,
        data_preprocessor.image_nan_to_num,
        data_preprocessor.labels_to_kast_streaks_artifact,
    ]
    data_preprocessor.set_pipeline(pipeline)
    params.update({param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor})

    model = DeepHiTSAtlas(params)
    train_set, val_set, test_set = model.get_dataset_used_for_training()
    test_pred = model.predict(test_set.data_array)
    plot_confusion_matrix(test_set.data_label, test_pred, show=True, title="Random")

    checkpoint_path = os.path.join(
        PROJECT_PATH,
        "results",
        "testing_atlas",
        "DeepHitsAtlas_20200627-201557",
        "checkpoints",
        "model",
    )
    model.load_model(checkpoint_path)
    # model.fit()
    test_pred = model.predict(test_set.data_array)
    plot_confusion_matrix(test_set.data_label, test_pred, show=True)

    blind_data = pd.read_pickle(os.path.join(PROJECT_PATH, "../ATLAS/atlas_blind.pkl"))
    blind_images = blind_data[general_keys.IMAGES]
    blind_features = blind_data[general_keys.FEATURES]
    blind_dataset = Dataset(blind_images, None, None, meta_data=blind_features)
    print(blind_dataset.data_array.shape)
    blind_dataset_preprop = model.dataset_preprocessor.preprocess_dataset(blind_dataset)
    print(blind_dataset_preprop.data_array.shape)
    blind_preds = model.predict(blind_dataset_preprop.data_array)
    plot_class_atlas_hist(blind_preds)
    plot_galactic_plane_of_blinds(blind_dataset_preprop, blind_preds)

    # a = pd.read_pickle(
    #     '../../tests/small_test_with_custome_features_and_metadata.pkl')
