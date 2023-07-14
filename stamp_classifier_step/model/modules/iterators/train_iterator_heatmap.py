from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules.iterators import train_iterator
from parameters import param_keys


class TrainIteratorHeatmap(train_iterator.TrainIteratorBuilder):
    def __init__(
        self,
        params,
        post_batch_processing=None,
        pre_batch_processing=None,
        drop_remainder=False,
    ):
        (
            self.dataset,
            self.sample_ph,
            self.heatmap_ph,
            self.label_ph,
        ) = self._create_dataset_with_placeholders(params)
        self.dataset = self._shuffle_and_repeat(
            self.dataset, params[param_keys.SHUFFLE_BUFFER_SIZE]
        )
        self.dataset = self._preprocess_batch(self.dataset, pre_batch_processing)
        self.dataset = self._batch_dataset(
            self.dataset, params[param_keys.BATCH_SIZE], drop_remainder
        )
        self.dataset = self._preprocess_batch(self.dataset, post_batch_processing)
        self.dataset = self._prefetch_batches(
            self.dataset, params[param_keys.PREFETCH_BUFFER_SIZE]
        )
        self.iterator = self._make_iterator(self.dataset)

    def _create_dataset_with_placeholders(self, params):
        sample_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(
                None,
                params[param_keys.INPUT_IMAGE_SIZE],
                params[param_keys.INPUT_IMAGE_SIZE],
                params[param_keys.N_INPUT_CHANNELS],
            ),
        )
        label_ph = tf.placeholder(dtype=tf.int64, shape=(None))
        heatmap_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(
                None,
                params[param_keys.INPUT_IMAGE_SIZE],
                params[param_keys.INPUT_IMAGE_SIZE],
            ),
        )
        dataset = tf.data.Dataset.from_tensor_slices((sample_ph, heatmap_ph, label_ph))
        return dataset, sample_ph, heatmap_ph, label_ph

    def get_placeholders(self):
        return self.sample_ph, self.heatmap_ph, self.label_ph

    def get_iterator_and_ph(self):
        return self.iterator, self.sample_ph, self.heatmap_ph, self.label_ph
