#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing ZTF database to be saved as a samplesx21x21x3 numpy array in a pickle 

TODO: clean_NaN once cropped
TODO: unit tests
ToDo: instead of cascade implement as pipeline, in order to have single call and definition
@author: asceta
"""
import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
import numpy as np


# comopse as a pipeline to choose preprocessing steps
class ZTFDataPreprocessorCascade(object):
    """
    Constructor
    """

    def __init__(self, dataset_obj: Dataset):
        self.dataset_obj = dataset_obj

    """
  define your preprocessing strategy here
  """

    def get_dataset(self):
        return self.dataset_obj

    def select_channels(self, channels_to_select):
        if len(self.dataset_obj.data_array.shape) == 1:
            raise ValueError(
                "Data array has shape %s, It's probabily due to misshapes in np.array(dataset). Try cleaning misshapes first"
            )
        selected_images_channels = self.dataset_obj.data_array[..., channels_to_select]
        if len(selected_images_channels.shape) == 3:
            selected_images_channels = selected_images_channels[..., np.newaxis]
        self.dataset_obj.data_array = selected_images_channels
        return self

    def normalize_by_sample(self):
        images = self.dataset_obj.data_array
        for image_index in range(images.shape[0]):
            image = images[image_index]
            image -= np.nanmin(image)
            image = image / np.nanmax(image)
            images[image_index] = image
        self.dataset_obj.data_array = images
        return self

    def normalize_by_image(self):
        images = self.dataset_obj.data_array
        for image_index in range(images.shape[0]):
            for channel_index in range(images.shape[-1]):
                image = images[image_index, ..., channel_index]
                image -= np.nanmin(image)
                image = image / np.nanmax(image)
                images[image_index, ..., channel_index] = image
        self.dataset_obj.data_array = images
        return self

    def nan_to_num(self, number_to_replace_nans):
        samples = self.dataset_obj.data_array
        nans_sample_idx = self._get_nans_samples_idx(samples)
        print(
            "%i samples with NaNs. NaNs replaced with number %s"
            % (len(nans_sample_idx), str(number_to_replace_nans))
        )
        samples[np.isnan(samples)] = number_to_replace_nans
        self.dataset_obj.data_array = samples
        return self

    def _check_all_removed(self, remove_name, samples_list, idxs_to_remove):
        if len(samples_list) == len(idxs_to_remove):
            raise OverflowError(
                "All samples have %s, thus batch is empty and cannot be processed"
                % remove_name
            )

    def _check_misshape_all_removed(self, samples_list, idxs_to_remove):
        self._check_all_removed("MISSHAPE", samples_list, idxs_to_remove)

    def _check_nan_all_removed(self, samples_list, idxs_to_remove):
        self._check_all_removed("NAN", samples_list, idxs_to_remove)

    def _get_misshaped_samples_idx(self, samples):
        miss_shaped_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if sample.shape[2] != 3 or sample.shape[1] != 63 or sample.shape[0] != 63:
                # print("sample %i of shape %s" % (i, str(sample.shape)))
                miss_shaped_sample_idx.append(i)
        self._check_misshape_all_removed(samples, miss_shaped_sample_idx)
        return miss_shaped_sample_idx

    def clean_misshaped(self):
        samples_clone = list(self.dataset_obj.data_array[:])
        labels_clone = list(self.dataset_obj.data_label[:])
        miss_shaped_sample_idx = self._get_misshaped_samples_idx(samples_clone)
        for index in sorted(miss_shaped_sample_idx, reverse=True):
            samples_clone.pop(index)
            labels_clone.pop(index)
        print(
            "%i misshaped samples removed. Remaining SNe %i"
            % (len(miss_shaped_sample_idx), int(np.sum(labels_clone)))
        )
        self.dataset_obj = Dataset(
            samples_clone, labels_clone, self.dataset_obj.batch_size
        )
        return self

    def _get_nans_samples_idx(self, samples):
        nans_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if np.isnan(sample).any():
                # print("sample %i of shape %s" %(i,str(sample.shape)))
                nans_sample_idx.append(i)
        self._check_nan_all_removed(samples, nans_sample_idx)
        return nans_sample_idx

    # TODO: refactor; fuse with clean misshaped
    def clean_nans(self):
        samples_clone = list(self.dataset_obj.data_array[:])
        labels_clone = list(self.dataset_obj.data_label[:])
        nans_sample_idx = self._get_nans_samples_idx(samples_clone)
        for index in sorted(nans_sample_idx, reverse=True):
            samples_clone.pop(index)
            labels_clone.pop(index)
        print(
            "%i samples with NaNs removed. Remaining SNe %i"
            % (len(nans_sample_idx), int(np.sum(labels_clone)))
        )
        self.dataset_obj = Dataset(
            samples_clone, labels_clone, self.dataset_obj.batch_size
        )
        return self


#
#   def crop_at_center(self, sample_numpy, cropsize=21):
#     center = int((sample_numpy.shape[1] - 1) / 2)
#     crop_side = int((cropsize - 1) / 2)
#     crop_begin = center - crop_side
#     crop_end = center + crop_side + 1
#     # print(center)
#     # print(crop_begin, crop_end)
#     return sample_numpy[:, crop_begin:crop_end, crop_begin:crop_end, :]
#
#   def clean_nans(self, dataset_obj: Dataset):
#     nans_sample_idx = self.get_nans_samples_idx(samples)
#     # print('%d samples with nans removed' %len(nans_sample_idx))
#     for index in sorted(nans_sample_idx, reverse=True):
#       samples.pop(index)
#     return samples
#
#   def zero_fill_nans(self, samples_numpy):
#     samples_with_nan_idx = []
#     for i in range(samples_numpy.shape[0]):
#       if (np.isnan(samples_numpy[i, ...]).any()):
#         samples_with_nan_idx.append(i)
#     # print('%d samples with NaNs (filled with 0s)' %len(samples_with_nan_idx))
#     return np.nan_to_num(samples_numpy)
#
#   def normalize_01(self, samples_numpy):
#     for i in range(samples_numpy.shape[0]):
#       for j in range(samples_numpy.shape[3]):
#         sample = samples_numpy[i, :, :, j]
#         normalized_sample = (sample - np.min(sample)) / np.max(
#             sample - np.min(sample))
#         samples_numpy[i, :, :, j] = normalized_sample
#     return samples_numpy
#
#   def print_sample(self, img):
#     fig = plt.figure()
#     for k, imstr in enumerate(['Template', 'Science', 'Difference']):
#       ax = fig.add_subplot(1, 3, k + 1)
#       ax.axis('off')
#       ax.set_title(imstr)
#       ax.matshow(img[..., k])
#
#
# if __name__ == "__main__":
#   path_data = '/home/ereyes/LRPpaper/datasets/ZTF'
#   path_reals = path_data + '/broker_reals.json'
#   path_bogus = path_data + '/broker_bogus.json'
#
#   data_processor = ZTF_data_preprocessor()
#
#   print('Number of reals: %d' % len(data_processor.json2list(path_reals)))
#   print('Number of bogus: %d' % len(data_processor.json2list(path_bogus)))
#
#   print("\nReals")
#   preprocessed_reals = data_processor.preprocesses_dataset(path_reals)
#   print("Bogus")
#   preprocessed_bogus = data_processor.preprocesses_dataset(path_bogus)
#
#   print(
#       '\nNumber of reals after preprocessing: %d' % preprocessed_reals.shape[0])
#   print('Number of bogus after preprocessing: %d' % preprocessed_bogus.shape[0])
#
#   data_processor.save_to_pickle(path_data, array2save=preprocessed_reals,
#                                 file_name="broker_reals")
#   data_processor.save_to_pickle(path_data, array2save=preprocessed_bogus,
#                                 file_name="broker_bogus")
