import os
import sys
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
import gzip
import base64
from astropy.io import fits
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(PROJECT_PATH)


def get_dict_with_first_epoch_with_byte_stamp(data_dict):
    dict_with_first_epoch = {}
    data_oids = list(data_dict.keys())
    objects_without_images = 0
    for oid in data_oids:
        if oid == "tar":
            continue
        print("analizing oid: %s ..." % str(oid))
        epoch_counter = 0
        epochs = list(data_dict[oid].keys())
        for epoch in epochs:
            data = data_dict[oid][epoch]
            if data is None:
                epoch_counter += 1
                continue
            else:
                dict_with_first_epoch[oid] = {"epoch": epoch, "stamps": data}
                break
        if oid not in dict_with_first_epoch:
            print("No images on oid: %s" % str(oid))
            objects_without_images += 1
    print("Object without images %i" % objects_without_images)  # 0 oids without images
    return dict_with_first_epoch


def save_pickle(dict, path):
    with open(path, "wb") as handle:
        pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def get_image_from_bytes_stamp(stamp_byte):
    with gzip.open(io.BytesIO(stamp_byte), "rb") as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data
    return img


def get_images_from_bytes_list(bytes_list):
    img_list = []
    for i, data in enumerate(bytes_list):
        byte_stamp = data["stampData"]
        img = get_image_from_bytes_stamp(byte_stamp)
        if img.shape[0] != 63 or img.shape[1] != 63:
            print(
                "image %i not square, shape (%i,%i)" % (i, img.shape[0], img.shape[1])
            )
        img_list.append(img)
    alert_imgs = np.stack(img_list, axis=-1)
    return alert_imgs


def get_dict_first_epoch_images(data_dict):
    dict_with_first_epoch = {}
    data_oids = list(data_dict.keys())
    objects_without_images = 0
    for i, oid in enumerate(data_oids):
        if oid == "tar":
            continue
        print("analizing %ith oid: %s ..." % (i, str(oid)))
        epoch_counter = 0
        epochs = list(data_dict[oid].keys())
        for epoch in epochs:
            data = data_dict[oid][epoch]
            if data is None:
                epoch_counter += 1
                continue
            else:
                dict_with_first_epoch[oid] = {
                    "epoch": epoch,
                    "byte_stamps": data,
                    "stamps": get_images_from_bytes_list(data),
                }
                break
        if oid not in dict_with_first_epoch:
            print("No images on oid: %s" % str(oid))
            objects_without_images += 1
    print("Object without images %i" % objects_without_images)  # 0 oids without images
    return dict_with_first_epoch


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH,
        "..",
        "datasets",
        "ZTF",
        "stamp_classifier",
        "filtered_stamps_by_label.pkl",
    )
    save_filtered_path = os.path.join(
        PROJECT_PATH,
        "..",
        "datasets",
        "ZTF",
        "stamp_classifier",
        "first_epoch_stamps.pkl",
    )
    data_dict = pd.read_pickle(data_path)
    data_dict_first_epoch = get_dict_first_epoch_images(data_dict)
    save_pickle(data_dict_first_epoch, save_filtered_path)
    oids = list(data_dict_first_epoch.keys())
    data_dict_first_epoch[oids[0]]["stamps"].shape
