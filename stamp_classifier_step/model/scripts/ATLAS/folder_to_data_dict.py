import os
import sys
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import csv

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.utils import save_pickle, check_path


if __name__ == "__main__":
    save_path = os.path.join(PROJECT_PATH, "..", "ATLAS")
    check_path(save_path)
    folder_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "01")
    dirs = [
        d
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]
    print(dirs)

    images = []
    labels_str = []
    labels_num = []
    only_numeric_metadata = []
    metadata = []
    # dir_idx = 0
    for dir_idx in range(len(dirs)):
        dir_name_i = dirs[dir_idx]
        specific_class_data_path = os.path.join(folder_path, dir_name_i)
        image_file_names = [
            f for f in os.listdir(specific_class_data_path) if f.endswith(".jpg")
        ]
        metadata_filename = [
            f for f in os.listdir(specific_class_data_path) if not f.endswith(".jpg")
        ]
        print(metadata_filename)
        if metadata_filename != []:
            metadata_filename = metadata_filename[0]
            metadata_path = os.path.join(specific_class_data_path, metadata_filename)
            # get metadata
            metadata_array = np.loadtxt(metadata_path, usecols=range(2, 75))
            print(metadata_array.shape)
            metadata_IDs = []
            for line in open(metadata_path):
                cols = line.split(" ")
                # print(cols)
                metadata_IDs.append(cols[0])
        # print(metadata_filename)

        # print(len(image_file_names))
        for image_name_i in image_file_names:
            image_oid = image_name_i.split(".fits.")[0]
            if dir_name_i == "streak":
                metadata_IDs = [
                    metadata_name.split(".")[0] for metadata_name in metadata_IDs
                ]
            # if image_oid has no feature, skip
            if len(np.argwhere(np.array(metadata_IDs) == image_oid)) == 0:
                print(image_oid, "not included beacause, it has no metadata")
                continue
            image_i_path = os.path.join(specific_class_data_path, image_name_i)
            # rgb
            img = mpimg.imread(image_i_path)
            if img.shape != (101, 101, 3):
                print(img.shape)
            images.append(img)
            labels_str.append(dir_name_i)
            labels_num.append(dir_idx)

            # print(image_oid)
            if metadata_filename != []:
                img_oid_idx_position_in_metadata_array = np.argwhere(
                    np.array(metadata_IDs) == image_oid
                )[0][0]
                image_metadata_ID = metadata_IDs[img_oid_idx_position_in_metadata_array]
                metadata_array_i = metadata_array[
                    img_oid_idx_position_in_metadata_array
                ]
                metadata_list = [image_metadata_ID] + list(metadata_array_i)
            else:
                metadata_array_i = np.empty((73,))
                metadata_array_i[:] = np.nan
            # print(metadata_array_i.shape)
            metadata_list = [image_metadata_ID] + list(metadata_array_i)
            # print(metadata_list)
            # print(image_oid)
            # print(image_name_i)
            # print(metadata_IDs[img_oid_idx_position_in_metadata_array])
            # print(metadata_array[img_oid_idx_position_in_metadata_array])
            # print(img_oid_idx_position_in_metadata_array)
            metadata.append(metadata_list)
            only_numeric_metadata.append(metadata_array_i)

    print(np.unique(labels_num, return_counts=True))
    print(np.array(images).shape)
    print(np.array(metadata).shape)

    data_dict = {
        "images": images,
        "labels": labels_num,
        "features": only_numeric_metadata,
        "labels_str": labels_str,
        "metadata_with_object_id": metadata,
    }

    save_pickle(data_dict, os.path.join(save_path, "atlas_data_with_metadata.pkl"))
    # plt.imshow(img)
    # plt.show()
