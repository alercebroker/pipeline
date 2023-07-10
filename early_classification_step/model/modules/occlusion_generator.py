# basic libraries
import os
import sys
import numpy as np
from copy import deepcopy

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)


class OcclusionGenerator(object):
    def __init__(self, img, box, step=1):
        """Initializations"""
        self.box = box
        self.box_size = box.shape[0]
        # self.img = img  #
        self.pad_size = self.box_size - 1
        self.img = self.pad_image(img, self.pad_size)
        self.img_size = self.img.shape[0]
        # print(self.img.shape)
        self.step = step
        self.i = 0
        self.j = 0

    def pad_image(self, image, pad_by):
        padded_image = np.pad(
            image, ((pad_by, pad_by), (pad_by, pad_by), (0, 0)), "constant"
        )
        return padded_image

    def flow(self):
        """Return a single occluded image and its location"""
        if self.i + self.box_size > self.img.shape[0]:
            return None, None, None

        retImg = np.copy(self.img)
        retImg[
            self.i : self.i + self.box_size, self.j : self.j + self.box_size, :
        ] = self.box

        old_i = deepcopy(self.i)
        old_j = deepcopy(self.j)

        # update indices
        self.j = self.j + self.step
        if self.j + self.box_size > self.img.shape[1]:  # reached end
            self.j = 0  # reset j
            self.i = self.i + self.step  # go to next row

        retImg_cut = retImg[
            self.pad_size : self.img_size - self.pad_size,
            self.pad_size : self.img_size - self.pad_size,
        ]
        # print(retImg_cut.shape)
        return retImg_cut, old_i, old_j

    def _get_max_batch_size(self):
        heigth, width, channels = self.img.shape
        max_batch_size = (heigth - self.box_size + 1) ** 2
        return max_batch_size

    def gen_minibatch(self, batchsize=None):
        """Returns a minibatch of images of size <=batchsize"""
        if batchsize is None:
            batchsize = self._get_max_batch_size()

        # list of occluded images
        occ_imlist = []
        locations = []
        for i in range(batchsize):
            occimg, i, j = self.flow()
            if occimg is not None:
                occ_imlist.append(occimg)
                locations.append([i, j])

        return np.array(occ_imlist), np.array(locations)


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from modules.data_loaders.frame_to_input import get_image_from_bytes_stamp
    import pandas as pd

    def df_to_array(df):
        removed_ids = []
        input_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        n_samples = len(df)
        data_array = []
        for i in range(n_samples):
            serie = df.iloc[i]
            image_array = []
            skip_image = False
            for key in input_keys:
                converted_image = get_image_from_bytes_stamp(serie[key])
                if converted_image.shape != (63, 63) or np.isnan(converted_image).any():
                    skip_image = True
                    removed_ids.append(i)
                    break
                else:
                    image_array.append(converted_image)
            if skip_image:
                skip_image = False
                continue
            image_tensor = np.stack(image_array, axis=2)
            data_array.append(image_tensor)
        data_array = np.stack(data_array, axis=0)
        return data_array

    input_df = pd.read_pickle(os.path.join(PROJECT_PATH, "tests/small_dataset.pkl"))
    input_data = df_to_array(input_df)

    input_image = input_data[np.random.randint(input_data.shape[0]), ..., -1][
        ..., np.newaxis
    ]

    # box_step = 20
    # box = np.zeros((img_size - box_step, img_size - box_step, 3))
    box = np.ones((6, 6, 1)) * np.nan
    occlusioner = OcclusionGenerator(input_image, box, step=6)  # int(box_step/2))

    (
        occluded_images,
        positions,
    ) = occlusioner.gen_minibatch()  # n samples: img_size + filter_size-1

    plt.rcParams["figure.figsize"] = (20, 20)
    for i in range(positions.shape[0]):
        print(i, "/", positions.shape[0])
        plt.subplot(
            int(np.sqrt(positions.shape[0])), int(np.sqrt(positions.shape[0])), i + 1
        )
        plt.imshow(occluded_images[i, ..., 0])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.imshow(input_image[..., 0])
    plt.axis("off")
    plt.show()
