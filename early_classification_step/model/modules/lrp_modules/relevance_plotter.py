#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
astronomical rotated image ploter

@author Esteban Reyes
"""

# python 2 and 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

# basic libraries
import numpy as np


class LRP_plot_tools(object):
    """
    Cosntructor
    """

    # def __init__(self, params=):

    def normalize_through_channels(self, batch):
        batch_without_padding = np.copy(batch[:, 3:24, 3:24, :])
        samples_max = np.max(
            np.max(np.max(np.abs(batch_without_padding), axis=1), axis=1), axis=1
        )
        for i in range(batch.shape[0]):
            batch_without_padding[i, :, :, :] = (
                batch_without_padding[i, :, :, :] / samples_max[i]
            )
        normalized_batch_center_in_05 = (batch_without_padding + 1.0) / 2.0
        return normalized_batch_center_in_05

    def normalize_through_all(self, batch):
        batch_without_padding = np.copy(batch[:, 3:24, 3:24, :])
        max = np.max(batch_without_padding)
        batch_without_padding = batch_without_padding / max
        normalized_batch_center_in_05 = (batch_without_padding + 1.0) / 2.0
        return normalized_batch_center_in_05

    def plot_sample(
        self, images, titles=None, vmin=None, vmax=None, cmap="gray", sup_title=None
    ):
        channels = images.shape[-1]
        # fill titles with blanks
        if titles == None:
            titles = []
            for i in range(channels):
                titles.append("")

        elif len(titles) < channels:
            for i in range(channels - len(titles)):
                titles.append("")

        fig, axs = plt.subplots(nrows=1, ncols=channels)
        if sup_title:
            st = fig.suptitle(sup_title)
        for i, ax in enumerate(axs):
            ax.imshow(
                images[..., i], interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.axis("off")
            ax.set_title(titles[i])
        fig.tight_layout()
        if sup_title:
            # st.set_y(0.95)
            fig.subplots_adjust(top=1.25)
            # pass
        plt.show()

    def plot_input_image(self, images, sup_title=None):
        self.plot_sample(
            images, ["Science", "Template", "Diff", "SNR Diff"], sup_title=sup_title
        )

    def plot_heatmap(self, images):
        self.plot_sample(images, vmax=1, vmin=0, cmap="jet")
