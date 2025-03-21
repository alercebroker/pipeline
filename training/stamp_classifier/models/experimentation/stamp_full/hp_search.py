import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import ray
from ray import tune
from ray.tune.stopper import Stopper
from ray.tune.search import ConcurrencyLimiter

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch


# new ray tune version:
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.search.hyperopt import HyperOptSearch

import numpy as np
import os
from typing import Dict
from collections import defaultdict

from data_loader import get_tf_datasets
from model import StampModelFull

PROJECT_DIR = ''


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
N_CLASSES = 6


def balanced_xentropy(labels, predicted_logits):
    predictions = tf.nn.softmax(predicted_logits).numpy()
    scores = []
    for class_index in range(predictions.shape[1]):
        objects_from_class = labels == class_index
        class_probs = predictions[objects_from_class, class_index]
        class_probs = np.clip(class_probs, 10**-15, 1 - 10**-15)
        class_score = - np.mean(np.log(class_probs))
        scores.append(class_score)
    return np.array(scores).mean()


def training_function(config):
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    dataset_name = config["dataset_name"]
    dropout_rate = config["dropout_rate"]
    with_batchnorm = config["with_batchnorm"]
    first_kernel_size = config["first_kernel_size"]

    layer_size_keys = sorted([k for k in config.keys() if 'layer_' in k])
    layer_sizes = [config[k] for k in layer_size_keys]

    with tf.device('/cpu:0'):
        training_dataset, validation_dataset, test_dataset, label_encoder = get_tf_datasets(
            dataset_name, batch_size=batch_size)

    stamps_shape = list(test_dataset.as_numpy_iterator())[0][0].shape[1:]
    if dataset_name == 'cropped' or dataset_name == 'low_res':
        stamp_classifier = StampClassifier16(
            stamps_shape, layer_sizes, dropout_rate,
            with_batchnorm, first_kernel_size, n_classes=N_CLASSES)
    elif dataset_name == 'multiscale':
        stamp_classifier = StampClassifierMultiScale(
            stamps_shape, n_levels=4, layer_sizes=layer_sizes,
            dropout_rate=dropout_rate, with_batchnorm=with_batchnorm,
            first_kernel_size=first_kernel_size, n_classes=N_CLASSES)
    elif dataset_name == 'full':
        stamp_classifier = StampModelFull(
            stamps_shape, layer_sizes, dropout_rate,
            with_batchnorm, first_kernel_size, n_classes=N_CLASSES)
    else:
        raise ValueError('hey!')

    for x, pos, y in test_dataset:
        _ = stamp_classifier((x[:10], pos[:10]))
        break

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function()
    def train_step(samples, positions, labels):
        with tf.GradientTape() as tape:
            predictions = stamp_classifier((samples, positions), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, stamp_classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, stamp_classifier.trainable_variables))

        # train_loss(loss)
        # train_accuracy(labels, predictions)

    # logdir = './logs/single_level_2'
    # train_writer = tf.summary.create_file_writer(logdir+'/train')
    # val_writer = tf.summary.create_file_writer(logdir+'/val')
    # test_writer = tf.summary.create_file_writer(logdir+'/test')

    def val_test_step(dataset, iteration, file_writer):
        prediction_list = []
        label_list = []
        for samples, pos, labels in dataset:
            predictions = stamp_classifier((samples, pos))
            prediction_list.append(predictions)
            label_list.append(labels)

        labels = tf.concat(label_list, axis=0)
        predictions = tf.concat(prediction_list, axis=0)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions.numpy().argmax(axis=1), average='macro')
        # xentropy_test = balanced_xentropy(labels, predictions)

        # with file_writer.as_default():
        #     tf.summary.scalar('precision', precision, step=iteration)
        #     tf.summary.scalar('recall', recall, step=iteration)
        #     tf.summary.scalar('f1', f1, step=iteration)
        #     tf.summary.scalar('loss', xentropy_test, step=iteration)

        return f1

    log_frequency = 50
    for iteration, training_batch in enumerate(training_dataset):
        x_batch, pos_batch, y_batch = training_batch
        train_step(x_batch, pos_batch, y_batch)
        # if iteration % 10 == 0 and iteration != 0:
        #     with train_writer.as_default():
        #         tf.summary.scalar('loss', train_loss.result(), step=iteration)
        #         tf.summary.scalar('accuracy', train_accuracy.result(), step=iteration)

        if iteration % 500 == 0:
            val_f1 = val_test_step(validation_dataset, iteration, None)
            tune.report(iterations=iteration, validation_f1=val_f1)
            # _ = val_test_step(test_dataset, iteration, test_writer)

        # Reset the metrics for the next iteration
        # train_loss.reset_states()
        # train_accuracy.reset_states()

    # train_writer.flush()
    # val_writer.flush()
    # test_writer.flush()


class NoImprovementStopper(Stopper):
    def __init__(self,
                 metric: str,
                 num_steps: int):
        self._metric = metric
        self.num_steps = num_steps
        self.historic_max = defaultdict(lambda: 0.0)
        self.steps_without_improvement = defaultdict(lambda: 0)

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)
        if metric_result > self.historic_max[trial_id]:
            self.historic_max[trial_id] = metric_result
            self.steps_without_improvement[trial_id] = 0
            return False
        else:
            self.steps_without_improvement[trial_id] += 1

        if self.steps_without_improvement[trial_id] >= self.num_steps:
            return True
        else:
            return False

    def stop_all(self):
        return False


if __name__ == '__main__':
    ray.init(configure_logging=False)

    algo = HyperOptSearch(n_initial_points=10)
    # algo = ConcurrencyLimiter(algo, max_concurrent=3)

    scheduler = AsyncHyperBandScheduler(
        time_attr="iterations", grace_period=5000, max_t=30000)

    hp_layer_sizes = {
        'cropped': {
            'layer_1': tune.lograndint(64, 192),
            'layer_2': tune.lograndint(8, 64),
            'layer_3': tune.lograndint(64, 160),
            'layer_4': tune.lograndint(48, 192),
            'layer_5': tune.lograndint(32, 160)
        },
        'low_res': {
            'layer_1': tune.lograndint(32, 194),
            'layer_2': tune.lograndint(8, 48),
            'layer_3': tune.lograndint(32, 194),
            'layer_4': tune.lograndint(8, 48),
            'layer_5': tune.lograndint(8, 32)
        },
        'full': {
            'layer_1': tune.lograndint(16, 96),
            'layer_2': tune.lograndint(32, 128),
            'layer_3': tune.lograndint(8, 64),
            'layer_4': tune.lograndint(8, 64),
            'layer_5': tune.lograndint(8, 48)
        },
        'multiscale': {
            'layer_1': tune.lograndint(8, 96),
            'layer_2': tune.lograndint(1, 64),
            'layer_3': tune.lograndint(32, 192),
            'layer_4': tune.lograndint(16, 96),
            'layer_5': tune.lograndint(32, 96)
        }
    }

    dataset_name = 'full'

    base_config = {
        "dataset_name": dataset_name,
        "batch_size": 256,
        "learning_rate": tune.loguniform(1e-4, 3e-3),
        "dropout_rate": tune.uniform(0.3, 0.95),
        "with_batchnorm": tune.choice([True, False]),
        "first_kernel_size": tune.choice([3, 5])
    }

    hp_config = {**base_config, **(hp_layer_sizes[dataset_name])}
    print(hp_config)

    analysis = tune.run(
        training_function,
        search_alg=algo,
        scheduler=scheduler,
        metric="validation_f1",
        mode="max",
        num_samples=200,
        config=hp_config,
        resources_per_trial={
            "cpu": 3,
            "gpu": 1
        },
        stop=NoImprovementStopper("validation_f1", 5),
        local_dir=PROJECT_DIR,
        name=dataset_name,
        # resume=True,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
