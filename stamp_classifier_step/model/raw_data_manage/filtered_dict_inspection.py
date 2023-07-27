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


def get_dict_with_first_epoch_with_images(data_dict):
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
        "first_epoch_byte_stamps.pkl",
    )
    print(os.path.abspath(data_path))
    data_dict = pd.read_pickle(data_path)
    print(data_dict.keys())
    # epochs = list(data_dict['ZTF19aaacrpc'].keys())
    # # for epoch in epochs:
    # #   print(type(data_dict['ZTF19aaacrpc'][epoch]))
    # #data_dict['tar'] #is just a list with strings
    # for epoch in epochs:
    #     data = data_dict['ZTF19aaacrpc'][epoch]
    #     if data is None:
    #       continue
    #     print(len(data_dict['ZTF19aaacrpc'][epoch]))
    data_oids = list(data_dict.keys())
    for oid in data_oids:
        if oid == "tar":
            continue
        print("analizing oid: %s ..." % str(oid))
        epochs = list(data_dict[oid].keys())
        for epoch in epochs:
            data = data_dict[oid][epoch]
            if data is None:
                continue
            if len(data) != 3:
                raise ValueError("len(data) != 3")

    # all stamps are from Science Template Difference
    data_dict_first_epoch = get_dict_with_first_epoch_with_images(data_dict)
    # save_pickle(data_dict_first_epoch, save_filtered_path)
    oids = list(data_dict_first_epoch.keys())
    img_byte_list = data_dict_first_epoch[oids[0]]["stamps"]
    # fig = plt.figure()
    for img_byte_i, imstr in enumerate(["Template", "Science", "Difference"]):
        stamp_byte = img_byte_list[img_byte_i]["stampData"]
        # ax = fig.add_subplot(1, 3, img_byte_i+1)
        # ax.axis('off')
        # ax.set_title(imstr)
        with gzip.open(io.BytesIO(stamp_byte), "rb") as f:
            with fits.open(io.BytesIO(f.read())) as hdul:
                img = hdul[0].data
                print(img.shape)
                print(img)
