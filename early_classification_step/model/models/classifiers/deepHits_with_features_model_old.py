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
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures
from parameters import param_keys, general_keys, errors
from modules.iterators.train_iterator_with_features import (
    TrainIteratorWithFeaturesBuilder,
)
from modules.iterators.validation_iterator_with_features import (
    ValidationIteratorWithFeaturesBuilder,
)
from modules.iterators.iterator_post_processing import augment_with_rotations_features
from modules.print_manager import PrintManager
from modules.data_set_generic import Dataset
from modules.networks.deep_hits_with_features import DeepHitsWithFeatures
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt


# TODO: refactor train test and validate
class DeepHiTSWithFeatures(DeepHiTSNanNormCropStampModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHitsWithFeatures", session=None):
        self.model_name = model_name
        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # TODO put this into an object; a param manager
        self.params = param_keys.default_params.copy()
        self._update_new_default_params()
        if params is not None:
            self.params.update(params)
        param_keys.update_paths_dict(self.params)

        (
            self.global_iterator,
            self.handle_ph,
            self.train_img_ph,
            self.train_feat_ph,
            self.train_lbl_ph,
            self.iterator_train,
            self.validation_img_ph,
            self.validation_feat_ph,
            self.validation_lbl_ph,
            self.iterator_validation,
        ) = self._iterator_init(self.params)

        (
            self.input_batch,
            self.input_features,
            self.input_labels,
            self.training_flag,
            self.logits,
            self.output_probabilities,
            self.output_pred_cls,
        ) = self._build_graph(self.params)

        self.loss = self._loss_init(
            self.logits, self.input_labels, self.params[param_keys.NUMBER_OF_CLASSES]
        )
        self.train_step, self.learning_rate = self._optimizer_init(
            self.loss, self.params[param_keys.LEARNING_RATE]
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
        self.metrics_dict = self._define_evaluation_metrics(
            self.input_labels, self.output_pred_cls
        )
        self.print_manager = PrintManager()
        # Init summaries
        self.summary_dict = self._init_summaries(self.loss, self.metrics_dict)

        self.data_loader, self.dataset_preprocessor = self._data_loader_init(
            self.params
        )

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
            param_keys.FEATURES_NAMES_LIST: [
                "sgscore1",
                "distpsnr1",
                "sgscore2",
                "distpsnr2",
                "sgscore3",
                "distpsnr3",
                "isdiffpos",
            ],
        }
        self.params.update(new_default_params)

    def _data_loader_init(self, params):
        data_loader = FrameToInputWithFeatures(params)
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

    # TODO implement builder pattern to avoid code replication and reduce 2 lines
    def _iterator_init(self, params):
        with tf.name_scope("iterators"):
            train_it_builder = TrainIteratorWithFeaturesBuilder(
                params, post_batch_processing=augment_with_rotations_features
            )
            (
                iterator_train,
                train_sample_ph,
                train_feature_ph,
                train_lbl_ph,
            ) = train_it_builder.get_iterator_and_ph()
            val_it_builder = ValidationIteratorWithFeaturesBuilder(
                params, post_batch_processing=augment_with_rotations_features
            )
            (
                iterator_val,
                val_sample_ph,
                val_feature_ph,
                val_lbl_ph,
            ) = val_it_builder.get_iterator_and_ph()
            handle_ph, global_iterator = train_it_builder.get_global_iterator()
        return (
            global_iterator,
            handle_ph,
            train_sample_ph,
            train_feature_ph,
            train_lbl_ph,
            iterator_train,
            val_sample_ph,
            val_feature_ph,
            val_lbl_ph,
            iterator_val,
        )

    def _define_inputs(self):
        input_batch, input_features, input_labels = self.global_iterator.get_next()
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        training_flag = tf.placeholder(tf.bool, shape=None, name="training_flag")
        return input_batch, input_features, input_labels, training_flag

    def _init_network(self, X, feature, params, training_flag):
        network = DeepHitsWithFeatures(X, feature, params, training_flag)
        return network.get_output()

    def _build_graph(self, params):
        with tf.name_scope("inputs"):
            (
                input_batch,
                input_features,
                input_labels,
                training_flag,
            ) = self._define_inputs()
        with tf.variable_scope("network"):
            logits = self._init_network(
                input_batch, input_features, params, training_flag
            )
        with tf.name_scope("outputs"):
            output_probabilities = tf.nn.softmax(logits)
            output_predicted_classes = tf.argmax(output_probabilities, axis=-1)
        return (
            input_batch,
            input_features,
            input_labels,
            training_flag,
            logits,
            output_probabilities,
            output_predicted_classes,
        )

    def _check_test_data_availability(self, test_data):
        if test_data:
            return Dataset(
                test_data[0],
                test_data[1],
                meta_data=test_data[2],
                batch_size=self.params[param_keys.BATCH_SIZE],
            )
        return None

    def _prepare_input(
        self, X, X_features, y, validation_data, test_data
    ) -> (Dataset, Dataset, Dataset):
        batch_size = self.params[param_keys.BATCH_SIZE]
        if (
            X.shape != ()
            and y.shape != ()
            and X_features.shape != ()
            and validation_data
        ):
            train_set = Dataset(X, y, meta_data=X_features, batch_size=batch_size)
            val_set = Dataset(
                validation_data[0],
                validation_data[1],
                meta_data=validation_data[2],
                batch_size=batch_size,
            )
            test_set = self._check_test_data_availability(test_data)

        elif (
            X.shape != ()
            and y.shape != ()
            and X_features.shape != ()
            and not validation_data
        ):
            aux_dataset_obj = Dataset(X, y, meta_data=X_features, batch_size=batch_size)
            data_divider = DatasetDividerInt(
                aux_dataset_obj,
                test_size=self.params[param_keys.TEST_SIZE],
                validation_size=self.params[param_keys.VAL_SIZE],
                val_random_seed=self.params[param_keys.VALIDATION_RANDOM_SEED],
            )
            train_set, val_set = data_divider.get_train_val_data_set_objs()
            test_set = self._check_test_data_availability(test_data)

        else:  # valdiation_data and test_data ignored
            train_set, val_set, test_set = self._data_init()
        train_set.balance_data_by_replication()
        train_set = self._global_shuffling(train_set)  # TODO: shuffle data
        return train_set, val_set, test_set

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
        self.best_model_so_far = {general_keys.ITERATION: 0, general_keys.LOSS: 1e10}
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
            self.val_set.data_array,
            self.val_set.meta_data,
            self.val_set.data_label,
            metrics,
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

    # get variables_list by evaluating data on all batches
    # TODO: flag for human readable dict, to avoid tensorf key
    def get_variables_by_batch(
        self, variables: list, data_array, features, labels=None
    ):
        if labels is None:
            labels = np.ones(data_array.shape[0])
        if not isinstance(variables, list):
            variables = [variables]  # BAD PRACTICE
        # Initialization of the iterator with the actual data
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: data_array,
                self.validation_feat_ph: features,
                self.validation_lbl_ph: labels,
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

    def evaluate_metrics(self, data_array, features, labels, metrics):
        metric_data = self.get_variables_by_batch(metrics, data_array, features, labels)
        # calculate mean of value per batch
        for metric in metrics:
            metric_mean = np.mean(metric_data[metric][general_keys.VALUES_PER_BATCH])
            metric_data[metric][general_keys.BATCHES_MEAN] = metric_mean
            errors.check_nan_metric(self.metrics_dict, metric, metric_mean)
        return metric_data

    def evaluate(self, data_array, features, data_label, set="Test"):
        metrics = list(self.summary_dict.keys())
        metric_data = self.evaluate_metrics(data_array, features, data_label, metrics)
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

    def _predict_template(self, data_array, feature, variable):
        predictions_dict = self.get_variables_by_batch(
            variables=[variable], data_array=data_array
        )
        predictions_by_batch_dict = predictions_dict[variable]
        predictions = np.concatenate(
            predictions_by_batch_dict[general_keys.VALUES_PER_BATCH]
        )
        return predictions

    def predict(self, data_array, features):
        return self._predict_template(data_array, features, self.output_pred_cls)

    def predict_proba(self, data_array, features):
        return self._predict_template(data_array, features, self.output_probabilities)

    # TODO: this should be a method inside Dataset class of train_set object
    def _global_shuffling(self, set: Dataset, seed=1234):
        idx_permuted = np.random.RandomState(seed=seed).permutation(
            set.data_array.shape[0]
        )
        set.data_array = set.data_array[idx_permuted]
        set.data_label = set.data_label[idx_permuted]
        set.meta_data = set.meta_data[idx_permuted]
        return set


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

    model = DeepHiTSWithFeatures(params)

    model.fit(
        X=np.ones((5000, 21, 21, 3)), X_features=np.ones((5000, 3)), y=np.ones((5000))
    )
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
