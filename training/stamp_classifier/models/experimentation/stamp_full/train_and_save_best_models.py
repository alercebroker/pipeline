import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import argparse
import joblib
import yaml
import os
import sys
import pickle

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from focal_loss import SparseCategoricalFocalLoss
from datetime import datetime

from data_loader import get_tf_datasets
from model import StampModelFull, StampModelModified

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
    def __init__(self, num_steps: int, mode: str = 'max'):
        self.num_steps = num_steps
        self.mode = mode
        self.historic_best = -float('inf') if mode == 'max' else float('inf')
        self.steps_without_improvement = 0

    def should_break(self, current_value):
        if (self.mode == 'max' and current_value > self.historic_best) or \
           (self.mode == 'min' and current_value < self.historic_best):
            self.historic_best = current_value
            self.steps_without_improvement = 0
            return False
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement >= self.num_steps


def eval_step(model, dataset):
    predictions, labels_list = [], []
    for samples, md, labels in dataset:
        logits = model((samples, md), training=False)
        predictions.append(logits)
        labels_list.append(labels)

    predictions = tf.concat(predictions, axis=0)
    labels = tf.concat(labels_list, axis=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), predictions.numpy().argmax(axis=1), average='macro')
    xentropy = balanced_xentropy(labels, predictions)

    return precision, recall, f1, xentropy, labels.numpy(), predictions.numpy().argmax(axis=1)


def save_confusion_matrix_and_report(labels, predictions, save_dir, class_names):
    os.makedirs(save_dir, exist_ok=True)

    # Original confusion matrix
    cm = confusion_matrix(labels, predictions, labels=class_names)
    _plot_confusion_matrix(cm, class_names, "Confusion Matrix", os.path.join(save_dir, "confusion_matrix.png"))

    # Normalized confusion matrix (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaNs in case of division by zero
    _plot_confusion_matrix(cm_normalized, class_names, "Normalized Confusion Matrix",
                           os.path.join(save_dir, "confusion_matrix_normalized.png"),
                           fmt=".2f")

    # Classification report
    report = classification_report(labels, predictions, target_names=class_names)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)


def _plot_confusion_matrix(cm, class_names, title, filepath, fmt='d'):
    num_classes = len(class_names)

    cell_size = 0.8
    width = max(6, cell_size * num_classes)
    height = max(5, cell_size * num_classes)
    plt.figure(figsize=(width, height))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})

    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(ha='right', fontsize=12, rotation='vertical')
    plt.yticks(fontsize=12, rotation='horizontal')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()



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
            gamma=2, from_logits=True,
            )
    else:
        print('Using standard crossentropy loss')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
            )
        
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

        return f1, xentropy_test

    stop_metric_type = args["stop_metric_type"] 
    stopper_mode = 'min' if stop_metric_type == 'loss' else 'max'

    log_frequency = 50
    stopper = NoImprovementStopper(num_steps=5, mode=stopper_mode)
    best_weights = None
    best_metric = float('inf') if stopper_mode == 'min' else -float('inf')

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
            val_f1, val_loss = val_test_step(validation_dataset, iteration, val_writer)
            val_metric = val_loss if stop_metric_type == 'loss' else val_f1
            val_test_step(test_dataset, iteration, test_writer)
            print(f"Validation F1-score at iteration {iteration}: {val_f1}")

            if (stopper_mode == 'min' and val_metric < best_metric) or (stopper_mode == 'max' and val_metric > best_metric):
                best_metric = val_metric
                best_weights = stamp_classifier.get_weights()

            if stopper.should_break(val_metric):
                break

        # Reset the metrics for the next iteration
        train_loss.reset_state()
        train_accuracy.reset_state()

    train_writer.flush()
    val_writer.flush()
    test_writer.flush()

    if best_weights is not None:
        stamp_classifier.set_weights(best_weights)
    
    _, _, _, _, test_labels, test_predictions = eval_step(stamp_classifier, test_dataset)
    test_labels = [dict_info_model['dict_mapping_classes'][x] for x in test_labels]
    test_predictions = [dict_info_model['dict_mapping_classes'][x] for x in test_predictions]
    class_names = list(dict_info_model['dict_mapping_classes'].values())
    save_confusion_matrix_and_report(test_labels, test_predictions, path_save_results, class_names=class_names)

    stamp_classifier.save(f"{path_save_results}/model.keras")

    name_qt_file = None
    if args_general['norm_type'] == 'QT':
        name_qt_file = 'quantile_transformer.pkl'
        joblib.dump(dict_info_model['qt'], f"{path_save_results}/quantile_transformer.pkl")

    output_dict = {
        'hp_model': hp_model,
        'args_general': args_general,
        'structure': {
            'order_features': dict_info_model['order_features'],
            'dict_mapping_classes': dict_info_model['dict_mapping_classes']
        },
        'normalization_stats': {
            'norm_means': dict_info_model['norm_means'],
            'norm_stds': dict_info_model['norm_stds'],
            'qt_name_file': name_qt_file
        }
    }

    with open(f"{path_save_results}/config_used.yaml", 'w') as f:
        yaml.dump(output_dict, f)


def parse_model_args(arg_dict=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # Argumentos generales
    parser.add_argument("--path_data", type=str,
                        default="./data/normalized_ndarrays_hasavro_2025-06-06_04-26.pkl")
    parser.add_argument("--results_folder", type=str,
                        default="./results_incremental_satbogus")
    parser.add_argument("--coord_type", type=str, default="cartesian",
                        choices=["cartesian", "spherical"],
                        help="Coordinate type to use")
    parser.add_argument("--norm_type", type=str, default="QT",
                        choices=["QT", "z-score", "none"],
                        help="Normalization method to apply")
    parser.add_argument("--stop_metric_type", type=str, default="loss", 
                        choices=["loss", "f1"],
                        help="Metric type used for early stopping (loss or f1)")

    parser.add_argument("--use_metadata", action='store_true')
    parser.add_argument("--add_new_sats_sn", action='store_true')

    # Hiperparámetros del modelo
    parser.add_argument("--stamp", type=str, default="full",
                        choices=["full", "modified"],
                        help="Model type to use")
    parser.add_argument("--bn", action='store_true')
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--focal_loss", action='store_true')

    args = parser.parse_args(None if arg_dict is None else [])
    return vars(args)

# ==== Main Script ====

args = parse_model_args()
results_folder = args["results_folder"]
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ROOT = os.path.join(results_folder, date)
os.makedirs(ROOT, exist_ok=True)

# Separar hp_model y args_general automáticamente
hp_model = {
    "model": args["stamp"],
    "with_batchnorm": args["bn"],
    "crop": args["crop"],
    "focal_loss": args["focal_loss"],
}

args_general = {
    "path_data": args["path_data"],
    "use_metadata": args["use_metadata"],
    "use_only_avro": "hasavro" in args["path_data"],
    "add_new_sats_sn": args["add_new_sats_sn"],
    "coord_type": args["coord_type"],
    "norm_type": args["norm_type"],
}

# Lógica para seleccionar configuración del modelo
if hp_model["model"] == "full":
    hp_model_base = hp_model_full  # Define esto en tu entorno
elif hp_model["model"] == "modified":
    hp_model_base = hp_model_modified  # Define esto también

# Mezclar los valores actualizados en hp_model_base
hp_model_base.update(hp_model)

# Entrenamiento
max_iterations = 10000000
num_runs = 1

for i in range(num_runs):
    print(f'run {i}/{num_runs-1}')
    train_model_and_save(hp_model_base, f'{ROOT}/run_{i}', args_general)

    #dict_config = {
    #    'hp_model': hp_model_base,
    #    'args_general': args_general
    #}

    #yaml_path = f'{ROOT}/run_{i}/config_used.yaml'
    #with open(yaml_path, 'w') as f:
    #    yaml.dump(dict_config, f, sort_keys=False)

    #print(f"Config saved to: {yaml_path}")