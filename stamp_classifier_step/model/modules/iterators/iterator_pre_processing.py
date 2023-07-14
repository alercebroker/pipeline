import tensorflow as tf
from parameters import param_keys


# ToDo: remove mishaped with tf while


class ZTFPreprocessorTf(object):
    def __init__(self, params):
        self.params = params
        self.channels_to_select = self.to_list(params[param_keys.CHANNELS_TO_USE])
        self.number_to_replace_nans = params[param_keys.NANS_TO]
        self.preprocessing_pipeline = self.get_default_pipeline()

    def get_default_pipeline(self):
        return [self.select_channels, self.normalize_by_sample]

    def to_list(self, variable):
        if type(variable) is not list:
            return [variable]
        return variable

    def preprocess_dataset(self, sample, label):
        for preprocessing_function in self.preprocessing_pipeline:
            sample, label = preprocessing_function(sample, label)
        return sample, label

    def append_to_pipeline(self, method):
        self.preprocessing_pipeline.append(method)
        return self

    def set_pipeline(self, pipeline):
        self.preprocessing_pipeline = pipeline

    def identity(self, sample, label):
        return sample, label

    # TODO: verify sample with 3 channels as input
    def select_channels(self, sample, label):
        channels = tf.split(sample, sample.shape[-1], -1)
        selected_channels_sample = tf.concat(
            [channels[i] for i in self.channels_to_select], axis=-1
        )
        return selected_channels_sample, label

    def normalize_by_sample(self, sample, label):
        sample_mean = tf.reduce_mean(tf.boolean_mask(sample, tf.is_nan(sample)))
        sample_nans_to_mean = tf.where(
            tf.is_nan(sample), tf.ones_like(sample) * sample_mean, sample
        )
        sample_nans_to_mean -= tf.reduce_min(sample_nans_to_mean)
        sample_nans_to_mean = sample_nans_to_mean / tf.reduce_max(sample_nans_to_mean)
        samples_norm_nan_to_num = tf.where(
            tf.is_nan(sample),
            tf.ones_like(sample) * self.number_to_replace_nans,
            sample_nans_to_mean,
        )
        return samples_norm_nan_to_num, label

    def normalize_by_image(self, sample, label):
        image_mean = tf.reshape(
            tf.reduce_mean(tf.boolean_mask(sample, tf.is_nan(sample)), axis=(0, 1)),
            [1, 1, -1],
        )
        sample_nans_to_mean = tf.where(
            tf.is_nan(sample), tf.ones_like(sample) * image_mean, sample
        )
        sample_nans_to_mean -= tf.reshape(
            tf.reduce_min(sample_nans_to_mean, axis=(0, 1)), [-1, 1, 1]
        )
        sample_nans_to_mean = sample_nans_to_mean / tf.reshape(
            tf.reduce_max(sample_nans_to_mean, axis=(0, 1)), [-1, 1, 1]
        )
        samples_norm_nan_to_num = tf.where(
            tf.is_nan(sample),
            tf.ones_like(sample) * self.number_to_replace_nans,
            sample_nans_to_mean,
        )
        return samples_norm_nan_to_num, label
