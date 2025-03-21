import tensorflow as tf
import numpy as np
import os
import sys

from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime

from data_loader import get_tf_datasets
from model import StampModelFull

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Best hyperparameters according to hp search
hp_model_full = {
    'layer_1': 37,
    'layer_2': 60,
    'layer_3': 29,
    'layer_4': 26,
    'layer_5': 31,
    'learning_rate': 0.0007589,
    'dropout_rate': 0.71726
}

# This hp are always the best ones
hp_model_full['with_batchnorm'] = True
hp_model_full['first_kernel_size'] = 3
hp_model_full['batch_size'] = 256


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


class NoImprovementStopper:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.historic_max = 0.0
        self.steps_without_improvement = 0

    def should_break(self, current_value):
        if current_value > self.historic_max:
            self.historic_max = current_value
            self.steps_without_improvement = 0
            return False
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.num_steps:
            return True
        else:
            return False


def train_model_and_save(hyperparameters, path_save_results):
    layer_size_keys = sorted([k for k in hyperparameters.keys() if 'layer_' in k])
    layer_sizes = [hyperparameters[k] for k in layer_size_keys]
    batch_size = hyperparameters['batch_size']
    dropout_rate = hyperparameters['dropout_rate']
    with_batchnorm = hyperparameters['with_batchnorm']
    first_kernel_size = hyperparameters['first_kernel_size']
    learning_rate = hyperparameters['learning_rate']

    with tf.device('/cpu:0'):
        training_dataset, validation_dataset, test_dataset, dict_mapping_classes \
            = get_tf_datasets(batch_size=batch_size)

    stamp_classifier = StampModelFull(
        layer_sizes=layer_sizes, 
        dropout_rate=dropout_rate,
        with_batchnorm=with_batchnorm, 
        first_kernel_size=first_kernel_size, 
        dict_mapping_classes=dict_mapping_classes)

    for x, pos, y in test_dataset:
        print(stamp_classifier((x[:10], pos[:10])))
        break

    for v in stamp_classifier.trainable_variables:
        print(v.name, v.shape, np.prod(v.shape))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
        )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False
        )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function()
    def train_step(samples, positions, labels):
        with tf.GradientTape() as tape:
            predictions = stamp_classifier((samples, positions), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, stamp_classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, stamp_classifier.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    logdir = f'{path_save_results}/logs'
    train_writer = tf.summary.create_file_writer(logdir + '/train')
    val_writer = tf.summary.create_file_writer(logdir + '/val')
    test_writer = tf.summary.create_file_writer(logdir + '/test')

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
        xentropy_test = balanced_xentropy(labels, predictions)

        with file_writer.as_default():
            tf.summary.scalar('precision', precision, step=iteration)
            tf.summary.scalar('recall', recall, step=iteration)
            tf.summary.scalar('f1', f1, step=iteration)
            tf.summary.scalar('loss', xentropy_test, step=iteration)

        return f1

    log_frequency = 50
    stopper = NoImprovementStopper(5)

    for iteration, training_batch in enumerate(training_dataset):
        x_batch, pos_batch, y_batch = training_batch
        train_step(x_batch, pos_batch, y_batch)
        if iteration % log_frequency == 0 and iteration != 0:
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=iteration)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=iteration)

        if iteration % 500 == 0:
            val_f1 = val_test_step(validation_dataset, iteration, val_writer)
            val_test_step(test_dataset, iteration, test_writer)
            print(f"Validation F1-score at iteration {iteration}: {val_f1}")
            if stopper.should_break(val_f1):
                break

        # Reset the metrics for the next iteration
        train_loss.reset_state()
        train_accuracy.reset_state()

    train_writer.flush()
    val_writer.flush()
    test_writer.flush()

    stamp_classifier.save(f"{path_save_results}/model.keras")

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = f'./results/{date}' 
os.makedirs(ROOT, exist_ok=True)

for i in range(5):
    print(f'run {i}/4')
    train_model_and_save(hp_model_full, f'{ROOT}/run_{i}')
