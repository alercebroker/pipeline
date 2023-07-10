#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unet with batchnormalization layers as in https://github.com/ankurhanda/tf-unet/blob/master/UNet.py
@author Esteban Reyes
"""

# python 2 and 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

# basic libraries
import os
import sys
import time

import numpy as np
import tensorflow as tf

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from modules.data_loaders.atlas_with_features_loader import ATLASwithFeaturesLoader
from parameters import param_keys, general_keys, errors
from modules.utils import save_pickle
import pandas as pd


# TODO: refactor train test and validate
class DeepHiTSAtlasWithFeatures(DeepHiTSWithFeaturesEntropyReg):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHiTSAtlasWithFeatures", session=None):
        super().__init__(params, model_name, session)

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.NUMBER_OF_CLASSES: 7,
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: None,
            param_keys.BATCHNORM_CONV: None,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 32,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.NANS_TO: 0,
            param_keys.INPUT_IMAGE_SIZE: 101,
            param_keys.CROP_SIZE: None,
            param_keys.ENTROPY_REG_BETA: 0.5,
            param_keys.INPUT_DATA_PREPROCESSOR: None,
        }
        self.params.update(new_default_params)

    def _data_loader_init(self, params):
        data_loader = ATLASwithFeaturesLoader(params)
        if params[param_keys.INPUT_DATA_PREPROCESSOR]:
            data_loader.dataset_preprocessor = params[
                param_keys.INPUT_DATA_PREPROCESSOR
            ]
        else:
            data_loader.dataset_preprocessor.set_pipeline(
                [
                    data_loader.dataset_preprocessor.image_check_single_image,
                    data_loader.dataset_preprocessor.image_select_channels,
                    data_loader.dataset_preprocessor.images_to_gray_scale,
                    data_loader.dataset_preprocessor.image_crop_at_center,
                    data_loader.dataset_preprocessor.image_normalize_by_image_1_1,
                    data_loader.dataset_preprocessor.image_nan_to_num,
                    data_loader.dataset_preprocessor.remove_streaks,
                    data_loader.dataset_preprocessor.features_normalize,
                ]
            )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor

    def fit(
        self,
        X=np.empty([]),
        X_features=np.empty([]),
        y=np.empty([]),
        validation_data=None,
        test_data=None,
        verbose=True,
        log_file="train.log",
    ):
        # create paths where training results will be saved
        self.create_paths()
        # verbose management TODO: put this inside a method
        print = self.print_manager.verbose_printing(verbose)
        file = open(os.path.join(self.model_path, log_file), "w")
        self.print_manager.file_printing(file)
        # init data
        self.train_set, self.val_set, self.test_set = self._prepare_input(
            X, X_features, y, validation_data, test_data
        )
        self.train_size = self.train_set.data_label.shape[0]
        # Initialization of the train iterator, only once since it is repeated forever
        self.sess.run(
            self.iterator_train.initializer,
            feed_dict={
                self.train_img_ph: self.train_set.data_array,
                self.train_feat_ph: self.train_set.meta_data,
                self.train_lbl_ph: self.train_set.data_label,
            },
        )
        # summaries TODO: put this inside a method
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.tb_path, "train"), self.sess.graph
        )
        self.val_writer = tf.summary.FileWriter(os.path.join(self.tb_path, "val"))
        merged = tf.summary.merge_all()

        print("\n", self.model_name)
        print("\n", self.params)
        print("\nBeginning training", flush=True)
        start_time = time.time()
        it = 0
        validation_monitor = self.params[param_keys.VALIDATION_MONITOR]
        self.best_model_so_far = {
            general_keys.ITERATION: 0,
            validation_monitor: self._get_validation_monitor_mode_value(
                self.params[param_keys.VALIDATION_MODE]
            ),
        }
        # model used to check if training iteration horizon should be increased
        self.it_horizon_increase_criterion_model = self.best_model_so_far.copy()
        # save first model
        self.saver.save(self.sess, self.checkpoint_path)
        # train model
        self._init_learning_rate()
        while it < self.params[param_keys.TRAIN_ITERATIONS_HORIZON]:

            # perform a trainning iteration
            if it % self.params[param_keys.PRINT_EVERY] != 0:
                self.sess.run(
                    self.train_step,
                    feed_dict={
                        self.handle_ph: self.train_handle,
                        self.training_flag: True,
                    },
                )
            else:
                # profiling ###############################
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                loss_value, summ, results_dict, _ = self.sess.run(
                    [self.loss, merged, self.metrics_dict, self.train_step],
                    feed_dict={
                        self.handle_ph: self.train_handle,
                        self.training_flag: True,
                    },
                )  # ,
                # summ_train_off = self.sess.run(merged, #monitor how variable behaive
                # when training_flag is False
                #     feed_dict={self.handle_ph: self.train_handle,
                #                self.training_flag: False})
                # self.val_writer.add_summary(summ_train_off, it)
                # options=run_options, run_metadata=run_metadata)
                # Train evaluation
                print(
                    "\n"
                    + self._get_message_to_print_loss_and_metrics(
                        previous_message="Epoch %i Iteration %i (train): Batch "
                        % (self._get_epoch(it, self.train_size, self.params), it),
                        loss_value=loss_value,
                        metrics_values_dict=results_dict,
                    ),
                    flush=True,
                )
                self.train_writer.add_summary(summ, it)
                # self.train_writer.add_run_metadata(run_metadata, 'step%d' % it)
            # check if validation must take place after every trainning iteration
            # and expand train horizon if validation criterion is met
            self.params[param_keys.TRAIN_ITERATIONS_HORIZON] = self._validate(it)
            self._check_learning_rate_update(it)
            if it % self.params[param_keys.PRINT_EVERY] == 0:
                time_usage = str(
                    datetime.timedelta(seconds=int(round(time.time() - start_time)))
                )
                print("Time usage: " + time_usage, flush=True)
            it += 1

        print("\nTraining Ended\n", flush=True)
        # Total time
        time_usage = str(
            datetime.timedelta(seconds=int(round(time.time() - start_time)))
        )
        print("Total training time: " + time_usage, flush=True)
        # restore best model so far
        self.saver.restore(self.sess, self.checkpoint_path)
        self.save_model()
        # evaluate final model
        print("\nEvaluating final model...", flush=True)
        print(
            "Best model @ it %d.\nValidation loss %.5f"
            % (
                self.best_model_so_far[general_keys.ITERATION],
                self.best_model_so_far[general_keys.LOSS],
            ),
            flush=True,
        )
        # final validation evaluation
        metrics = self.evaluate(
            self.val_set.data_array,
            self.val_set.meta_data,
            self.val_set.data_label,
            set="Validation",
        )
        # evaluate model over test set
        if self.test_set:  # if it is not None
            metrics = self.evaluate(
                self.test_set.data_array,
                self.test_set.meta_data,
                self.test_set.data_label,
                set="Test",
            )
        # closing train.log, printers and writers
        self.print_manager.close()
        file.close()
        self.train_writer.close()
        self.val_writer.close()
        return metrics

    def save_model(self, model_folder_path=None):
        """save model and and feature_normalization_statistics"""
        if self.dataset_preprocessor.feature_normalization_stats_dict is None:
            print("Forcefully initializing feature normalization stats")
            self._prepare_input()
        # print(self.dataset_preprocessor.feature_normalization_stats_dict)
        if model_folder_path is None:
            model_folder_path = os.path.join(self.model_path)
        save_pickle(
            self.dataset_preprocessor.feature_normalization_stats_dict,
            os.path.join(model_folder_path, "feature_norm_stats.pkl"),
        )
        # d = pd.read_pickle(os.path.join(path, "feature_norm_stats.pkl"))
        # print(d)
        self.saver.save(
            self.sess, os.path.join(model_folder_path, "checkpoints", "model")
        )

    def load_model_and_feature_stats(self, model_folder_path):
        d = pd.read_pickle(os.path.join(model_folder_path, "feature_norm_stats.pkl"))
        # print(d)
        self.dataset_preprocessor.feature_normalization_stats_dict = d
        self.saver.restore(
            self.sess, os.path.join(model_folder_path, "checkpoints", "model")
        )


if __name__ == "__main__":
    n_classes = 5
    params = {
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
    }

    model = DeepHiTSWithFeaturesEntropyReg(params)

    model.fit(
        X=np.ones((5000, 21, 21, 3)), X_features=np.ones((5000, 3)), y=np.ones((5000))
    )
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
