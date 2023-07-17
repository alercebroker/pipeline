from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules.iterators.train_iterator_heatmap import TrainIteratorHeatmap


class ValidationIteratorHeatmap(TrainIteratorHeatmap):
    def __init__(
        self,
        params,
        post_batch_processing=None,
        pre_batch_processing=None,
        drop_remainder=False,
    ):
        super().__init__(
            params, post_batch_processing, pre_batch_processing, drop_remainder
        )

    def _shuffle_and_repeat(self, dataset, shuffle_buffer):
        return dataset
