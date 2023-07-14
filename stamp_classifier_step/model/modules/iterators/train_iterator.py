from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parameters import param_keys


# ToDo : explore idea to give parameter as placeholder to varying size in
# validation. Think is not possible, because batch size is defined when val
# iterator is created
class TrainIteratorBuilder(object):
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
        dataset = tf.data.Dataset.from_tensor_slices((sample_ph, label_ph))
        return dataset, sample_ph, label_ph

    def _shuffle_and_repeat(self, dataset, shuffle_buffer):
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.repeat()
        return dataset

    def _batch_dataset(self, dataset, batch_size, drop_remainder):
        return dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    def _preprocess_batch(self, dataset, preprocessing_function):
        if preprocessing_function is None:
            return dataset
        else:
            return dataset.map(preprocessing_function)

    def _prefetch_batches(self, dataset, prefetch_buffer):
        return dataset.prefetch(buffer_size=prefetch_buffer)

    def _make_iterator(self, dataset):
        return tf.compat.v1.data.make_initializable_iterator(dataset)

    def get_global_iterator(self):
        handle_ph = tf.compat.v1.placeholder(tf.string, shape=[])
        global_iterator = tf.compat.v1.data.Iterator.from_string_handle(
            handle_ph,
            tf.compat.v1.data.get_output_types(self.dataset),
            tf.compat.v1.data.get_output_shapes(self.dataset),
        )
        return handle_ph, global_iterator

    def get_placeholders(self):
        return self.sample_ph, self.label_ph

    def get_iterator(self):
        return self.iterator

    def get_iterator_and_ph(self):
        return self.iterator, self.sample_ph, self.label_ph
