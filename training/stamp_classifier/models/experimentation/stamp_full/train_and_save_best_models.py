import tensorflow as tf
import numpy as np
import yaml
import os
import sys
import pickle

from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime

from data_loader import get_tf_datasets
from model import StampModelFull
from model import StampModelModified

import argparse
from focal_loss import SparseCategoricalFocalLoss


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

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

hp_model_modified = {
    'layer_1': 32,
    'layer_2': 32,
    'layer_3': 64,
    'layer_4': 64,
    'layer_5': 64,
    'layer_6': 31,
    'learning_rate': 0.0007589,
    'dropout_rate': 0.71726
}

# This hp are always the best ones
hp_model_full['first_kernel_size'] = 3
hp_model_full['batch_size'] = 256

hp_model_modified['first_kernel_size'] = 4
hp_model_modified['batch_size'] = 256


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


def train_model_and_save(hyperparameters, path_save_results, args_general):
    layer_size_keys = sorted([k for k in hyperparameters.keys() if 'layer_' in k])
    layer_sizes = [hyperparameters[k] for k in layer_size_keys]
    batch_size = hyperparameters['batch_size']
    dropout_rate = hyperparameters['dropout_rate']
    with_batchnorm = hyperparameters['with_batchnorm']
    first_kernel_size = hyperparameters['first_kernel_size']
    learning_rate = hyperparameters['learning_rate']
    crop = hyperparameters['crop']
    model_ = hyperparameters['model']
    focal_loss = hyperparameters['focal_loss']

    with tf.device('/cpu:0'):
        training_dataset, validation_dataset, test_dataset, dict_info_model \
            = get_tf_datasets(batch_size, args_general)

    if model_ == 'full':
        print('Using full model')
        stamp_classifier = StampModelFull(
            layer_sizes=layer_sizes, 
            dropout_rate=dropout_rate,
            with_batchnorm=with_batchnorm, 
            first_kernel_size=first_kernel_size,
            with_crop=crop, 
            dict_mapping_classes=dict_info_model['dict_mapping_classes'],
            order_features=dict_info_model['order_features'],
            norm_means=dict_info_model['norm_means'],
            norm_stds=dict_info_model['norm_stds'],
            )
    else:
        print('Using modified model')
        stamp_classifier = StampModelModified(
            layer_sizes=layer_sizes, 
            dropout_rate=dropout_rate,
            with_batchnorm=with_batchnorm, 
            first_kernel_size=first_kernel_size,
            with_crop = crop, 
            dict_mapping_classes=dict_info_model['dict_mapping_classes'],
            order_features=dict_info_model['order_features'],
            norm_means=dict_info_model['norm_means'],
            norm_stds=dict_info_model['norm_stds'],
            )

    for x, md, y in test_dataset:
        print(stamp_classifier((x[:10], md[:10])))
        break

    for v in stamp_classifier.trainable_variables:
        print(v.name, v.shape, np.prod(v.shape))

    if focal_loss:
        print('Using focal loss')
        loss_object = SparseCategoricalFocalLoss(
            gamma=2,from_logits=True,
            )
    else:
        print('Using standard crossentropy loss')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False
        )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function()
    def train_step(samples, metadata, labels):
        with tf.GradientTape() as tape:
            predictions = stamp_classifier((samples, metadata), training=True)
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
        for samples, md, labels in dataset:
            predictions = stamp_classifier((samples, md))
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
        if iteration >= max_iterations:
            print(f"Reached max_iterations = {max_iterations}, stopping training.")
            break

        x_batch, md_batch, y_batch = training_batch
        train_step(x_batch, md_batch, y_batch)
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
    
    str_ = ''
    if args_general['use_metadata']:
        str_ += '_md'
    if focal_loss:
        str_ += '_focal'
    if with_batchnorm:
        str_ += '_bn'
    if crop:
        str_ += '_crop'  
    if args_general['use_only_avro']:
        str_ += '_crop'    

    stamp_classifier.save(f"{path_save_results}/model_{model_}{str_}.keras")


date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = f'./results_incremental/{date}' 
os.makedirs(ROOT, exist_ok=True)

def parse_model_args(arg_dict=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    #parser.add_argument("--path_data", type=str, default="./data/normalized_ndarrays_2025-06-06_04-04.pkl")
    parser.add_argument("--path_data", type=str, default="./data/normalized_ndarrays_hasavro_2025-06-06_04-26.pkl")
    parser.add_argument("--stamp", type=str, default="full")
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--crop", type=bool, default=False)
    parser.add_argument("--use_metadata", type=bool, default=True)
    parser.add_argument("--focal_loss", type=bool, default=False)

    args = parser.parse_args(None if arg_dict is None else [])

    return args

max_iterations = 100000000
num_runs = 1
for i in range(num_runs):
    print(f'run {i}/{num_runs-1}')
    args = vars(parse_model_args(arg_dict=None))
    #print(args)
    model_ = args['stamp']
    if model_ == 'full':
        hp_model = hp_model_full
    elif model_ == 'modified':
        hp_model = hp_model_modified
    hp_model["with_batchnorm"] = args['bn']
    hp_model["crop"] = args['crop']
    hp_model["model"] = model_
    hp_model["focal_loss"] = args['focal_loss']

    args_general = {
        "path_data": args["path_data"],
        "use_metadata": args["use_metadata"],
        "use_only_avro": args["path_data"].find('hasavro') != -1
    }

    train_model_and_save(hp_model, f'{ROOT}/run_{i}', args_general)


    dict_config = {
        'hp_model': hp_model,
        'args_general': args_general
    }
    yaml_path = f'{ROOT}/run_{i}/config_used.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dict_config, f, sort_keys=False)

    print(f"Config saved to: {yaml_path}")