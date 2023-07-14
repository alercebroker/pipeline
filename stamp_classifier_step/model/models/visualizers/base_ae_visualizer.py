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

# basic libraries
import os
import sys
import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib

matplotlib.use("agg")

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_n_samples_heatmaps import (
    DatasetDividerIntHeatmaps,
)
from modules.print_manager import PrintManager
from modules import metrics
from modules.iterators.train_iterator_heatmap import TrainIteratorHeatmap
from modules.iterators.validation_iterator_heatmap import ValidationIteratorHeatmap
from modules import losses, optimizers
from modules.data_loaders.ztf_heatmap_loader import HeatmapLoader
from parameters import param_keys, general_keys, errors, constants
from modules.networks.deep_hits import DeepHits
from modules.networks.deep_hits_conv_decoder import DeepHitsConvDecoder

# import modules.layers as layers
from models.classifiers.base_model import BaseModel
from modules.iterators.iterator_post_processing import (
    augment_with_rotations_hm,
    augment_with_rotations_single,
)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid


# TODO: FIX!!! when you pass X, y in fit, y is heatmaps and not labels, which is missleading


# TODO: refactor train test and validate
class BaseAEVisualizer(BaseModel):
    """
    Constructor
    """

    def __init__(
        self,  # model_to_visualize,
        params=None,
        model_name="BaseAEVisualizer",
        session=None,
    ):
        # with tf.variable_scope("model2vis"):
        #   self.model_to_visualize = model_to_visualize

        self.model_name = model_name

        self.params = param_keys.default_params.copy()
        self._update_new_default_params()
        if params is not None:
            self.params.update(params)

        (
            self.global_iterator,
            self.handle_ph,
            self.train_img_ph,
            self.train_hm_ph,
            self.train_lbl_ph,
            self.iterator_train,
            self.validation_img_ph,
            self.validation_hm_ph,
            self.validation_lbl_ph,
            self.iterator_validation,
        ) = self._iterator_init(self.params)

        (
            self.input_batch,
            self.input_heatmap,
            self.input_labels,
            self.training_flag,
            self.output,
        ) = self._build_graph(self.params)

        self.loss = self._loss_init(self.input_heatmap, self.output)
        self.train_step, self.learning_rate = self._optimizer_init(
            self.params, self.loss, constants.DECODER
        )

        self.sess = self._session_init(session)
        self.saver = tf.train.Saver()
        self._variables_init()

        # Initialization of handles
        self.train_handle, self.validation_handle = self.sess.run(
            [
                self.iterator_train.string_handle(),
                self.iterator_validation.string_handle(),
            ]
        )
        self.metrics_dict = self._define_evaluation_metrics()
        self.print_manager = PrintManager()
        # Init summaries
        self.summary_dict = self._init_summaries(self.loss, self.metrics_dict)

        self.data_loader, self.dataset_preprocessor = self._data_loader_init(
            self.params
        )

        self.img_summary_dict = self._init_image_summaries(
            self.input_heatmap, self.output
        )

    # Todo: refactor visual
    def create_paths(self):
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_path = os.path.join(
            PROJECT_PATH,
            "results",
            self.params[param_keys.RESULTS_FOLDER_NAME],
            "%s_%s" % (self.model_name, date),
        )
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints", "model")
        self.tb_path = os.path.join(self.model_path, "tb_summaries")
        self.visual_path = os.path.join(self.model_path, "visual")
        if not os.path.exists(self.visual_path):
            os.makedirs(self.visual_path)

    # Todo: refactor
    def _get_merged_summ(self):
        summ = []
        for var in tf.get_collection(tf.GraphKeys.SUMMARIES):
            if "img" in var.name:
                continue
            summ.append(var)
        return tf.summary.merge(summ)

    def _init_image_summaries(self, label_hm, output_hm, n_images=6):
        expanded_lbl_hm = tf.expand_dims(label_hm, axis=-1)
        label_hm_img_summ = tf.summary.image(
            "label_hm_img", expanded_lbl_hm, max_outputs=n_images
        )
        output_hm_img_summ = tf.summary.image(
            "output_hm_img", output_hm, max_outputs=n_images
        )
        return {
            general_keys.LABEL_HEATMAP: label_hm_img_summ,
            general_keys.OUTPUT_HEATMAP: output_hm_img_summ,
            general_keys.MERGED_IMAGE_SUMM: tf.summary.merge(
                [label_hm_img_summ, output_hm_img_summ]
            ),
        }

    def _first_summary_image(self):
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: self.fixed_batch_data,
                self.validation_hm_ph: self.fixed_batch_hm,
                self.validation_lbl_ph: np.zeros_like(self.fixed_batch_hm),
            },
        )
        summ = self.sess.run(
            self.img_summary_dict[general_keys.MERGED_IMAGE_SUMM],
            feed_dict={
                self.handle_ph: self.validation_handle,
                self.training_flag: False,
            },
        )
        self.val_writer.add_summary(summ, 0)

    def _summary_output_heatmap(self, iteration):
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: self.fixed_batch_data,
                self.validation_hm_ph: np.zeros_like(self.fixed_batch_hm),
                self.validation_lbl_ph: np.zeros_like(self.fixed_batch_hm),
            },
        )
        summ = self.sess.run(
            self.img_summary_dict[general_keys.OUTPUT_HEATMAP],
            feed_dict={
                self.handle_ph: self.validation_handle,
                self.training_flag: False,
            },
        )
        self.val_writer.add_summary(summ, iteration)

    # TODO: merge with  _summary_output_heatmap
    def _matplot_output_heatmap(self, iteration):
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: self.fixed_batch_data,
                self.validation_hm_ph: self.fixed_batch_hm,
                self.validation_lbl_ph: np.zeros_like(self.fixed_batch_hm),
            },
        )
        gen_hm = self.sess.run(
            self.output,
            feed_dict={
                self.handle_ph: self.validation_handle,
                self.training_flag: False,
            },
        )

        heatmaps_stacked = np.stack(
            [self.fixed_batch_data[..., -1], self.fixed_batch_hm, gen_hm[..., 0]],
            axis=0,
        )
        n_samples = 10
        selected_heatmaps = heatmaps_stacked[:, :n_samples, ...]

        scale = 1.8
        fig = plt.figure(figsize=(3 * scale, n_samples * scale))
        grid = Grid(
            fig,
            rect=111,
            nrows_ncols=(n_samples, 3),
            axes_pad=0.2,
            label_mode="L",
            share_x=True,
            share_y=True,
        )

        for n, ax in enumerate(grid):
            sample_i = n % n_samples
            image_i = n % 3
            ax.imshow(selected_heatmaps[image_i, sample_i, ...])
            if n == 0:
                ax.set_title(r"Input Image", fontsize=10)
                ax.set_ylabel("sample %i" % 0)
            if n == 1:
                ax.set_title(r"Label Heatmap", fontsize=10)
            if n == 2:
                ax.set_title(r"Estimated Heatmap", fontsize=10)
            if n != 0 and n % 3 == 0:
                ax.set_ylabel("sample %i" % int(n / 3), fontsize=10)
            ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
        fig.savefig(
            os.path.join(self.visual_path, "hms_%i.png" % iteration),
            bbox_inches="tight",
        )
        plt.close()

    def _data_loader_init(self, params):
        data_loader = HeatmapLoader(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [
                data_loader.dataset_preprocessor.image_check_single_image,
                data_loader.dataset_preprocessor.image_clean_misshaped,
                data_loader.dataset_preprocessor.image_select_channels,
                data_loader.dataset_preprocessor.image_normalize_by_image,
                data_loader.dataset_preprocessor.image_nan_to_num,
                data_loader.dataset_preprocessor.image_crop_at_center,
            ]
        )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.NUMBER_OF_CLASSES: 5,
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: None,
            param_keys.BATCHNORM_CONV: None,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 50,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.NANS_TO: 0,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.CROP_SIZE: 21,
        }
        self.params.update(new_default_params)

    def _define_evaluation_metrics(self):
        return {}

    def _check_test_data_availability(self, test_data):
        if test_data:
            return Dataset(
                test_data[0], test_data[1], self.params[param_keys.BATCH_SIZE]
            )
        return None

    def _prepare_input(self, X, y, validation_data, test_data):
        if X.shape != () and y.shape != () and validation_data:
            train_set = Dataset(X, y, self.params[param_keys.BATCH_SIZE])
            val_set = Dataset(
                validation_data[0],
                validation_data[1],
                self.params[param_keys.BATCH_SIZE],
            )
            test_set = self._check_test_data_availability(test_data)

        elif X.shape != () and y.shape != () and not validation_data:
            aux_dataset_obj = Dataset(X, y, self.params[param_keys.BATCH_SIZE])
            data_divider = DatasetDividerIntHeatmaps(
                aux_dataset_obj,
                validation_size=self.params[param_keys.VAL_SIZE],
                val_random_seed=self.params[param_keys.VALIDATION_RANDOM_SEED],
            )
            train_set, val_set = data_divider.get_train_val_data_set_objs()
            test_set = self._check_test_data_availability(test_data)

        else:  # caldiation_data and test_data ignored
            train_set, val_set, test_set = self._data_init()
        train_set = self._global_shuffling(train_set)
        # print(train_set.data_array.shape,val_set.data_array.shape)
        # print(train_set.data_label.shape, val_set.data_label.shape)
        self.fixed_batch_data = val_set.data_array[: self.params[param_keys.BATCH_SIZE]]
        self.fixed_batch_hm = val_set.data_label[: self.params[param_keys.BATCH_SIZE]]
        return train_set, val_set, test_set

    # TODO: this should be a method inside Dataset class of train_set object
    def _global_shuffling(self, set, seed=1234):
        data_array = set.data_array
        data_label = set.data_label
        idx_permuted = np.random.RandomState(seed=seed).permutation(data_array.shape[0])
        data_array = data_array[idx_permuted, ...]
        data_label = data_label[idx_permuted, ...]
        set.data_array = data_array
        set.data_label = data_label
        return set

    # def create_patched_images(self, image_array):
    #   # not good idea (too much data)
    #   self.patch_size = 7
    #   n_images, heigth, width, channels = image_array.shape
    #   for image_indx in range(n_images):
    #     patched_images = []
    #     for i in range(heigth - self.patch_size + 1):
    #       for j in range(width - self.patch_size + 1):
    #         None
    #
    # def create_heatmaps(self, X, y):
    #   # not good idea (too much data)
    #   None

    def _get_epoch(self, it, train_size, params):
        batch_size = params[param_keys.BATCH_SIZE]
        n_examples_seen = it * batch_size
        return n_examples_seen // train_size

    # TODO: maybe create model trainer class
    # TODO: add method metadata_runner_profiling
    # TODO: come up with an easy way to select best model by loss, accuracy or
    #  something else
    # TODO: manage epochs in a smarter way
    def fit(
        self,
        X=np.empty([]),
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
            X, y, validation_data, test_data
        )
        self.train_size = self.train_set.data_label.shape[0]
        # Initialization of the train iterator, only once since it is repeated forever
        self.sess.run(
            self.iterator_train.initializer,
            feed_dict={
                self.train_img_ph: self.train_set.data_array,
                self.train_hm_ph: self.train_set.data_label,
                self.train_lbl_ph: np.zeros_like(self.train_set.data_label),
            },
        )
        # summaries TODO: put this inside a method
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.tb_path, "train"), self.sess.graph
        )
        self.val_writer = tf.summary.FileWriter(os.path.join(self.tb_path, "val"))
        # merged = tf.summary.merge_all()
        merged = self._get_merged_summ()

        print("\n", self.model_name)
        print("\n", self.params)
        print("\nBeginning training", flush=True)
        start_time = time.time()
        it = 0
        self.best_model_so_far = {general_keys.ITERATION: 0, general_keys.LOSS: 1e10}
        # model used to check if training iteration horizon should be increased
        self.it_horizon_increase_criterion_model = self.best_model_so_far.copy()
        # save first model
        self.saver.save(self.sess, self.checkpoint_path)
        self._first_summary_image()
        self._matplot_output_heatmap(0)
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
                    self._get_message_to_print_loss_and_metrics(
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
            self.val_set.data_array, self.val_set.data_label, set="Validation"
        )
        # evaluate model over test set
        if self.test_set:  # if it is not None
            metrics = self.evaluate(
                self.test_set.data_array, self.test_set.data_label, set="Test"
            )
        # closing train.log, printers and writers
        self.print_manager.close()
        file.close()
        self.train_writer.close()
        self.val_writer.close()
        return metrics

    # TODO: implement as validate method from keras
    # def test_over_multiple_data_sets(self, data_set_paths_list, set='test',
    #     verbose=True):
    #   if data_set_paths_list is None:
    #     return
    #   for data_set_path in data_set_paths_list:
    #     test_imgs, test_masks = self._get_set_images_and_labels(
    #         data_set_path, set)
    #     print("\n\n-------\n%s Metrics for dataset: %s" % (set, data_set_path))
    #     self.test(test_imgs, test_masks, set=set, verbose=verbose)

    def _get_message_to_print_loss_and_metrics(
        self, previous_message, loss_value, metrics_values_dict
    ):
        message = previous_message + "loss %f" % loss_value
        for metric_name in metrics_values_dict.keys():
            message += ", %s %f" % (metric_name, metrics_values_dict[metric_name])
        return message

    def _check_learning_rate_update(self, it):
        if self.params[param_keys.ITERATIONS_TO_UPDATE_LEARNING_RATE] == 0:
            return
        if it % self.params[param_keys.ITERATIONS_TO_UPDATE_LEARNING_RATE] == 0:
            self.update_learning_rate(it)
            print(
                "\n[PARAM UPDATE] Iteration %i (train): Learning rate updated: %.4f\n"
                % (it, self.sess.run(self.learning_rate)),
                flush=True,
            )

    # TODO: avoid assign added to graph at every run
    def _init_learning_rate(self):
        self.sess.run(
            tf.assign(self.learning_rate, self.params[param_keys.LEARNING_RATE])
        )

    # TODO: avoid assign added to graph at every run
    def update_learning_rate(self, global_step):
        self.sess.run(
            tf.assign(
                self.learning_rate,
                self.params[param_keys.LEARNING_RATE]
                / (
                    2.0
                    ** (
                        global_step
                        // self.params[param_keys.ITERATIONS_TO_UPDATE_LEARNING_RATE]
                    )
                ),
            )
        )

    # TODO: [MUST prec and rec are not batch calculable] make metric estimation
    #  more robust to avoid Nans, by calculation over al rate
    #  counts instead of batch
    def evaluate_metrics(self, data_array, labels, metrics):
        metric_data = self.get_variables_by_batch(metrics, data_array, labels)
        # calculate mean of value per batch
        for metric in metrics:
            metric_mean = np.mean(metric_data[metric][general_keys.VALUES_PER_BATCH])
            metric_data[metric][general_keys.BATCHES_MEAN] = metric_mean
            errors.check_nan_metric(self.metrics_dict, metric, metric_mean)
        return metric_data

    # get variables_list by evaluating data on all batches
    # TODO: flag for human readable dict, to avoid tensorf key
    def get_variables_by_batch(self, variables: list, data_array, labels=None):
        if labels is None:
            labels = np.ones(data_array.shape[0])
        if not isinstance(variables, list):
            variables = [variables]  # BAD PRACTICE
        # Initialization of the iterator with the actual data
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: data_array,
                self.validation_hm_ph: labels,
                self.validation_lbl_ph: np.zeros_like(labels),
            },
        )
        # dict of variables data to retrieve
        # TODO: dict creatin as methos, or maybe evaluation as class
        variables_data = {}
        for variable in variables:
            variables_data[variable] = {general_keys.VALUES_PER_BATCH: []}
        # evaluate set.
        while True:
            try:
                # get batch value
                variables_value = self.sess.run(
                    variables,
                    feed_dict={
                        self.handle_ph: self.validation_handle,
                        self.training_flag: False,
                    },
                )
                # append every batch metric to metric_data
                for variable, value in zip(variables, variables_value):
                    variables_data[variable][general_keys.VALUES_PER_BATCH].append(
                        value
                    )
            except tf.errors.OutOfRangeError:
                break
        return variables_data

    # TODO: include preprocessings and set default to false or false in training
    # TODO: metric_data should be returned with names paired or at least be a list
    def evaluate(self, data_array, data_label, set="Test"):
        metrics = list(self.summary_dict.keys())
        metric_data = self.evaluate_metrics(data_array, data_label, metrics)
        results_dict = self._pair_metric_names_and_mean_values_in_dict(
            metrics_dict=self.metrics_dict, metric_mean_values_dict=metric_data
        )
        print(
            self._get_message_to_print_loss_and_metrics(
                previous_message="\n%s Metrics: " % set,
                loss_value=metric_data[self.loss][general_keys.BATCHES_MEAN],
                metrics_values_dict=results_dict,
            ),
            flush=True,
        )
        results_dict.update(
            {general_keys.LOSS: metric_data[self.loss][general_keys.BATCHES_MEAN]}
        )
        return results_dict

    def _predict_template(self, data_array, variable):  # , preprocess_data):
        # if preprocess_data:
        #   preprocessed_dataset = self.dataset_preprocessor.preprocess_dataset(
        #       Dataset(data_array, None, None))
        #   data_array = preprocessed_dataset.data_array
        predictions_dict = self.get_variables_by_batch(
            variables=[variable], data_array=data_array
        )
        predictions_by_batch_dict = predictions_dict[variable]
        predictions = np.concatenate(
            predictions_by_batch_dict[general_keys.VALUES_PER_BATCH]
        )
        return predictions

    def predict(self, data_array):  # , preprocess_data=True):
        return self._predict_template(data_array, self.output_pred_cls)  # ,
        # preprocess_data)

    def predict_proba(self, data_array):  # , preprocess_data=True):
        return self._predict_template(data_array, self.output_probabilities)  # ,
        # preprocess_data)

    def _pair_metric_names_and_mean_values_in_dict(
        self, metrics_dict, metric_mean_values_dict
    ):
        paired_metric_name_mean_value_dict = {}
        for metric_name in metrics_dict.keys():
            paired_metric_name_mean_value_dict[metric_name] = metric_mean_values_dict[
                metrics_dict[metric_name]
            ][general_keys.BATCHES_MEAN]
        return paired_metric_name_mean_value_dict

    def _check_first_epoch_wait(self, current_iteration, train_size, params):
        if not params[param_keys.WAIT_FIRST_EPOCH]:
            return False
        return self._get_epoch(current_iteration, train_size, params) == 0

    # TODO: improve verbose management in validation, to make it salint if wanted,
    #  by just turning print verbose to false it wont be true after validation
    # TODO: add 1 to train_horizon, to capture last evaluation
    # TODO: improve how best model refresh criterion is selcted, know its done by
    #  hand
    def _validate(self, current_iteration):
        # check if current_iteration is multiple of VALIDATION_PERIOD
        # to perform validation every VALIDATION_PERIOD iterations
        # To perform first validation, an epoch must have past
        if current_iteration % self.params[
            param_keys.ITERATIONS_TO_VALIDATE
        ] != 0 or self._check_first_epoch_wait(
            current_iteration, self.train_size, self.params
        ):
            return self.params[param_keys.TRAIN_ITERATIONS_HORIZON]
        # variables to store: validation accuracy and loss
        metrics = list(self.summary_dict.keys())
        metric_data = self.evaluate_metrics(
            self.val_set.data_array, self.val_set.data_label, metrics
        )
        # write to validation summary all evaluated variables
        # TODO optimize summary evaluation by generating dict and run over
        #  all metrics at once
        for metric in metrics:
            summ = self.sess.run(
                self.summary_dict[metric],
                feed_dict={metric: metric_data[metric][general_keys.BATCHES_MEAN]},
            )
            self.val_writer.add_summary(summ, current_iteration)
        # print results TODO: manage result printing in a class
        results_dict = self._pair_metric_names_and_mean_values_in_dict(
            metrics_dict=self.metrics_dict, metric_mean_values_dict=metric_data
        )
        print(
            self._get_message_to_print_loss_and_metrics(
                previous_message="Epoch %i Iteration %i (val): "
                % (
                    self._get_epoch(current_iteration, self.train_size, self.params),
                    current_iteration,
                ),
                loss_value=metric_data[self.loss][general_keys.BATCHES_MEAN],
                metrics_values_dict=results_dict,
            ),
            flush=True,
        )
        # eval criterion to save best model and keep training
        criterion = metric_data[self.loss][general_keys.BATCHES_MEAN]
        # Check if criterion is met to refresh best model so far and overwrite
        # model checkpoint
        if criterion < self.best_model_so_far[general_keys.LOSS]:
            self.best_model_so_far[general_keys.LOSS] = criterion
            self.best_model_so_far[general_keys.ITERATION] = current_iteration
            print(
                "\nNew best validation model: Loss %.4f @ it %d\n"
                % (
                    self.best_model_so_far[general_keys.LOSS],
                    self.best_model_so_far[general_keys.ITERATION],
                ),
                flush=True,
            )
            self.saver.save(self.sess, self.checkpoint_path)
            self._summary_output_heatmap(current_iteration)
            self._matplot_output_heatmap(current_iteration)
        # check train horizon extention criterion, which is:
        # if current model loss is under CRITERION_%*loss of
        # last model that met criterion, train horizon is extended
        if (
            criterion
            < self.params[param_keys.CRITERION_PERCENTAGE]
            * self.it_horizon_increase_criterion_model[general_keys.LOSS]
        ):
            self.it_horizon_increase_criterion_model[general_keys.LOSS] = criterion
            self.it_horizon_increase_criterion_model[
                general_keys.ITERATION
            ] = current_iteration
            new_train_horizon = (
                current_iteration + self.params[param_keys.TRAIN_HORIZON_INCREMENT]
            )
            if new_train_horizon > self.params[param_keys.TRAIN_ITERATIONS_HORIZON]:
                print(
                    "Train iterations increased to %d because of model with loss %.4f @ it %d\n"
                    % (
                        new_train_horizon,
                        self.it_horizon_increase_criterion_model[general_keys.LOSS],
                        self.it_horizon_increase_criterion_model[
                            general_keys.ITERATION
                        ],
                    ),
                    flush=True,
                )
                return new_train_horizon

        return self.params[param_keys.TRAIN_ITERATIONS_HORIZON]

    # TODO implement builder pattern to avoid code replication and reduce 2 lines
    def _iterator_init(self, params):
        with tf.name_scope("iterators"):
            train_it_builder = TrainIteratorHeatmap(
                params, post_batch_processing=augment_with_rotations_hm
            )
            (
                iterator_train,
                train_sample_ph,
                train_heatmap_ph,
                train_lbl_ph,
            ) = train_it_builder.get_iterator_and_ph()
            val_it_builder = ValidationIteratorHeatmap(
                params, post_batch_processing=augment_with_rotations_hm
            )
            (
                iterator_val,
                val_sample_ph,
                val_heatmap_ph,
                val_lbl_ph,
            ) = val_it_builder.get_iterator_and_ph()
            handle_ph, global_iterator = train_it_builder.get_global_iterator()
        return (
            global_iterator,
            handle_ph,
            train_sample_ph,
            train_heatmap_ph,
            train_lbl_ph,
            iterator_train,
            val_sample_ph,
            val_heatmap_ph,
            val_lbl_ph,
            iterator_val,
        )

    def _define_inputs(self):
        input_batch, input_heatmaps, input_labels = self.global_iterator.get_next()
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        training_flag = tf.placeholder(tf.bool, shape=None, name="training_flag")
        return input_batch, input_heatmaps, input_labels, training_flag

    def _init_encoder(self, X, params, training_flag):
        network = DeepHits(X, params, training_flag)
        return network.get_last_conv_layer()

    def _init_decoder(self, X, params, training_flag):
        network = DeepHitsConvDecoder(X, params, training_flag)
        return network.get_output()

    def _loss_init(self, true_heatmap, output_heatmaps):
        with tf.name_scope("loss"):
            loss = losses.reconstruction_error(
                true_heatmap, tf.squeeze(output_heatmaps)
            )
        return loss

    def _optimizer_init(self, params, loss, scope_name):
        learning_rate_value = params[param_keys.LEARNING_RATE]
        with tf.name_scope("%s_optimizer" % scope_name):
            train_step, learning_rate = optimizers.adam(
                loss, learning_rate_value, scope_name=scope_name
            )
        tf.summary.scalar("%s_learning_rate" % scope_name, learning_rate)

        return train_step, learning_rate

    def _build_graph(self, params):
        with tf.name_scope("inputs"):
            (
                input_batch,
                input_heatmaps,
                input_labels,
                training_flag,
            ) = self._define_inputs()
        with tf.name_scope("encoder"):
            with tf.variable_scope("network"):
                encoded_input = self._init_encoder(input_batch, params, training_flag)
        # enc = tf.Print(encoded_input, [encoded_input, tf.shape(encoded_input)])
        with tf.variable_scope("decoder"):
            output = self._init_decoder(encoded_input, params, training_flag)
        return input_batch, input_heatmaps, input_labels, training_flag, output

    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    def load_encoder(self, path):
        aux_saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")
        )
        aux_saver.restore(self.sess, path)


if __name__ == "__main__":
    model = BaseAEVisualizer()
    checkpoint_path = os.path.join(
        PROJECT_PATH, "results/best_model_so_far/checkpoints", "model"
    )
    model.load_encoder(checkpoint_path)
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
