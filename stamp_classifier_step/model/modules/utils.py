import csv
import os
import pickle as pkl
import sys
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# matplotlib.use('Agg')

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from parameters import general_keys
from scipy import stats


def get_pvalue_welchs_ttest(list_value_1, list_value_2, show_print=False):
    p_value = stats.ttest_ind(list_value_1, list_value_2, equal_var=False)[1]
    if show_print:
        print("Welch’s t_test p_values: ", p_value)
    return p_value


def get_folder_names_in_path(path):
    return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]


def createCircularMask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask * 1.0


def plot_n_images(
    dataset, name, save_path, plot_show=False, n=100, set_to_plot=general_keys.TRAIN
):
    all_imgs = dataset[set_to_plot].data_array  #
    n_imags_available = all_imgs.shape[0]
    random_img_idxs = np.random.choice(range(n_imags_available), n)
    imgs = all_imgs[random_img_idxs, :, :, 0]
    sqrt_n = int(np.sqrt(n))
    fig, axs = plt.subplots(
        sqrt_n, sqrt_n, figsize=(16, 16), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    fig.suptitle(name, fontsize=40, color="white")
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, "%s.png" % name))
    if plot_show:
        plt.show()


def save_2d_image(
    image, name, save_folder=None, plot_show=False, img_size=16, axis_show="off"
):
    fig, ax = plt.subplots(1, 1, figsize=(img_size, img_size))
    fig.suptitle(name, fontsize=40, color="white")
    ax.imshow(image)
    ax.axis(axis_show)
    fig.tight_layout()
    if save_folder:
        fig.savefig(os.path.join(save_folder, "%s.png" % name))
    if plot_show:
        plt.show()


def plot_image(image, name=None, path=None, show=True):
    # fill titles with blanks
    titles = ["template", "science", "difference", "SNR difference"]
    n_channels = image.shape[-1]
    for i in range(n_channels):
        plt.subplot(1, 4, i + 1)
        if i == 0:
            indx = 1
        elif i == 1:
            indx = 0
        else:
            indx = i
        plt.imshow(image[:, :, indx], interpolation="nearest")
        plt.axis("off")
        plt.title(titles[i], fontdict={"fontsize": 15})
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.1)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path or name:
        plt.savefig(
            os.path.join(PROJECT_PATH, "figure_creation/figs/%s.svg" % name),
            format="svg",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    if show:
        plt.show()


def check_paths(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def check_path(path):
    check_paths(path)


def merge_datasets_dict(datasets_dict1, datasets_dict2):
    merged_datasets_dict = {}
    for set in datasets_dict1.keys():
        data_array = np.concatenate(
            [datasets_dict1[set].data_array, datasets_dict2[set].data_array]
        )
        data_label = np.concatenate(
            [datasets_dict1[set].data_label, datasets_dict2[set].data_label]
        )
        merged_datasets_dict[set] = Dataset(data_array, data_label, batch_size=50)
    return merged_datasets_dict


def save_pickle(data, path):
    with open(path, "wb") as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


def add_text_to_beginning_of_file_path(file_path, added_text):
    """add text to beginning of file name, without modifying the base path of the original file"""
    folder_path = os.path.dirname(file_path)
    data_file_name = os.path.basename(file_path)
    converted_data_path = os.path.join(
        folder_path, "%s_%s" % (added_text, data_file_name)
    )
    return converted_data_path


def set_soft_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def normalize_sum1(array, axis=-1):
    sums = np.sum(array, axis=axis)
    return array / np.expand_dims(sums, axis)


def delta_timer(delta_time):
    hours, rem = divmod(delta_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def create_auc_table(path, metric="roc_auc"):
    file_path = glob(os.path.join(path, "*", "*.npz"))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    methods = set()
    for p in file_path:
        _, f_name = os.path.split(p)
        dataset_name, method, single_class_name = f_name.split(sep="_")[:3]
        methods.add(method)
        npz = np.load(p)
        roc_auc = npz[metric]
        results[dataset_name][single_class_name][method].append(roc_auc)

    for ds_name in results:
        for sc_name in results[ds_name]:
            for method_name in results[ds_name][sc_name]:
                roc_aucs = results[ds_name][sc_name][method_name]
                print(method_name, " ", roc_aucs)
                results[ds_name][sc_name][method_name] = [
                    np.mean(roc_aucs),
                    0 if len(roc_aucs) == 1 else np.std(np.array(roc_aucs)),
                ]

    with open(os.path.join(path, "results-{}.csv".format(metric)), "w") as csvfile:
        fieldnames = ["dataset", "single class name"] + sorted(list(methods))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ds_name in sorted(results.keys()):
            for sc_name in sorted(results[ds_name].keys()):
                row_dict = {"dataset": ds_name, "single class name": sc_name}
                row_dict.update(
                    {
                        method_name: "{:.5f} ({:.5f})".format(
                            *results[ds_name][sc_name][method_name]
                        )
                        for method_name in results[ds_name][sc_name]
                    }
                )
                writer.writerow(row_dict)
