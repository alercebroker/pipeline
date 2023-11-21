from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
import os
import pickle

from sklearn.metrics import precision_recall_fscore_support

from lc_classifier.base import AstroObject
from .base import all_features_from_astro_objects
from .base import Classifier, NotTrainedException
from .preprocess import MLPFeaturePreprocessor


class ZTFClassifier(Classifier):
    version = '1.0.0'

    def __init__(self, list_of_classes: List[str]):
        self.list_of_classes = list_of_classes
        self.preprocessor = MLPFeaturePreprocessor()
        self.full_model = None
        self.inference_model = None
        self.feature_list = None

    def classify_single_object(self, astro_object: AstroObject) -> None:
        self.classify_batch([astro_object])

    def classify_batch(
            self,
            astro_objects: List[AstroObject],
            return_dataframe: bool = False) -> Optional[pd.DataFrame]:

        if self.feature_list is None or self.inference_model is None:
            raise NotTrainedException(
                'This classifier is not trained or has not been loaded')
        features_df = all_features_from_astro_objects(astro_objects)
        features_df = features_df[self.feature_list]
        features_np = self.preprocessor.preprocess_features(
            features_df).values
        probs_np = self.inference_model.inference(features_np).numpy()
        for object_probs, astro_object in zip(probs_np, astro_objects):
            data = np.stack(
                [
                    self.list_of_classes,
                    object_probs.flatten()
                ],
                axis=-1
            )
            object_probs_df = pd.DataFrame(
                data=data,
                columns=[['name', 'value']]
            )
            object_probs_df['fid'] = None
            object_probs_df['sid'] = 'ztf'
            object_probs_df['version'] = self.version
            astro_object.predictions = object_probs_df

        if return_dataframe:
            dataframe = pd.DataFrame(
                data=probs_np,
                columns=self.list_of_classes,
                index=features_df.index
            )
            return dataframe

    def fit(
            self,
            astro_objects: List[AstroObject],
            labels: pd.DataFrame,
            config: Dict):
        assert len(astro_objects) == len(labels)
        all_features_df = all_features_from_astro_objects(astro_objects)
        self.fit_from_features(all_features_df, labels, config)

    def fit_from_features(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            config: Dict):
        self.feature_list = features.columns.values
        training_labels = labels[labels['partition'] == 'training']
        training_features = features.loc[training_labels['aid'].values]
        self.preprocessor.fit(training_features)
        training_features = self.preprocessor.preprocess_features(
            training_features)

        validation_labels = labels[labels['partition'] == 'validation']
        validation_features = features.loc[validation_labels['aid'].values]
        validation_features = self.preprocessor.preprocess_features(
            validation_features)

        self.full_model = MLPModel(
            self.list_of_classes,
            config['learning_rate'],
            config['batch_size']
        )

        self.full_model.train(
            training_features, training_labels,
            validation_features, validation_labels)

    def save_classifier(self, directory: str):
        if self.full_model is None:
            raise NotTrainedException(
                'Cannot save model that has not been trained')

        if not os.path.exists(directory):
            os.mkdir(directory)

        self.preprocessor.save(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )

        with open(
                os.path.join(
                    directory,
                    'features_and_classes.pkl'),
                'wb') as f:
            pickle.dump(
                {
                    'feature_list': self.feature_list,
                    'list_of_classes': self.list_of_classes
                },
                f
            )

        export_archive = tf.keras.export.ExportArchive()
        export_archive.track(self.full_model)
        n_features = len(self.feature_list)
        export_archive.add_endpoint(
            name="inference",
            fn=lambda x: self.full_model.call(x, training=False, logits=False),
            input_signature=[
                tf.TensorSpec(
                    shape=(None, n_features),
                    dtype=tf.float32
                )
            ]
        )
        export_archive.write_out(
            os.path.join(
                directory,
                "tf_model"
            )
        )

    def load_classifier(self, directory: str):
        self.preprocessor.load(
            os.path.join(
                directory,
                'preprocessor.pkl'
            )
        )

        features_and_classes = pd.read_pickle(
            os.path.join(
                directory,
                'features_and_classes.pkl'
            )
        )
        self.feature_list = features_and_classes['feature_list']
        self.list_of_classes = features_and_classes['list_of_classes']

        self.inference_model = tf.saved_model.load(
            os.path.join(
                directory,
                "tf_model"
            )
        )


class MLPModel(tf.keras.Model):
    def __init__(
            self,
            list_of_classes: List[str],
            learning_rate: float,
            batch_size: int
    ):
        super().__init__()
        self.list_of_classes = list_of_classes
        self.batch_size = batch_size

        # simulate missing features
        self.input_dropout = tf.keras.layers.Dropout(0.2)
        self.dense_layer_1 = Dense(
            1_000,
            name='dense_layer_1',
        )
        self.dropout_between_layers = tf.keras.layers.Dropout(0.5)
        self.dense_layer_2 = Dense(
            1_000,
            name='dense_layer_2',
        )
        self.dense_layer_3 = Dense(
            len(self.list_of_classes),
            name='dense_layer_3'
        )
        self.activation = tf.nn.relu

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=25000,
            warmup_target=learning_rate,
            warmup_steps=1500,
            alpha=5e-2
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        self.loss_computer = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.max_validations_without_improvement = 3

    def call(self, x, training=False, logits=True):
        x = self.input_dropout(x, training=training)
        x = self.dense_layer_1(x)
        x = self.activation(x)
        x = self.dropout_between_layers(x, training=training)
        x = self.dense_layer_2(x)
        x = self.activation(x)
        x = self.dense_layer_3(x)

        if logits:
            return x
        else:
            return tf.nn.softmax(x)

    @tf.function
    def train_step(
            self,
            x_batch: tf.Tensor,
            y_batch: tf.Tensor):

        with tf.GradientTape() as tape:
            y_pred = self.__call__(x_batch, training=True, logits=True)
            loss_value = self.loss_computer(y_batch, y_pred)

            loss_value = loss_value \
                         + tf.keras.regularizers.L1(l1=1e-5)(self.dense_layer_1.kernel) \
                         + tf.keras.regularizers.L1(l1=1e-5)(self.dense_layer_2.kernel)

        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def validation_or_test_step(self, dataset: tf.data.Dataset) -> Tuple[np.array, np.array]:
        predictions = []
        labels = []
        for x_batch, y_batch in dataset:
            y_pred = self.__call__(
                x_batch, training=False, logits=False)
            labels.append(y_batch.numpy())
            predictions.append(y_pred.numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return labels, predictions

    def _compute_stats(
            self,
            labels: np.array,
            predicted_probabilities: np.array) -> Tuple[float, float, float]:

        labels = [label.decode('UTF-8') for label in labels]
        predictions = self.list_of_classes[np.argmax(predicted_probabilities, axis=1)]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro')
        return precision, recall, f1

    def train(
            self,
            training_features: pd.DataFrame,
            training_labels: pd.DataFrame,
            validation_features: pd.DataFrame,
            validation_labels: pd.DataFrame):

        training_dataset = self._tf_dataset_from_dataframes(
            training_features, training_labels, rebalance=True)
        validation_dataset = self._tf_dataset_from_dataframes(
            validation_features, validation_labels, rebalance=False)

        iteration = 0

        self.validations_without_improvement = 0
        self.best_historic_loss = np.inf
        for x_batch, y_batch in training_dataset:
            training_loss = self.train_step(x_batch, y_batch)
            iteration += 1
            if iteration % 250 == 0:
                print(
                    'iteration', iteration,
                    'training loss', f'{training_loss.numpy():.3f}',
                    f'lr {self.optimizer.learning_rate.numpy():.3e}')

            if iteration % 2500 == 0:
                val_labels, val_predictions = self.validation_or_test_step(
                    validation_dataset)
                val_precision, val_recall, val_f1 = self._compute_stats(
                    val_labels, val_predictions)
                print(f'iteration {iteration} valstats f1 {val_f1:.3f} '
                      f'precision {val_precision:.3f} recall {val_recall:.3f}')

                should_stop = self._evaluate_training_stopper(-val_f1)  # minus because it expects a loss
                if should_stop:
                    break

    def _tf_dataset_from_dataframes(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            rebalance: bool
    ):

        if rebalance:
            # Note: balanced dataset is infinite
            labels = self._label_list_to_one_hot(labels['astro_class'].values)
            datasets_from_class = []
            for class_i in range(len(self.list_of_classes)):
                from_class = labels[:, class_i].astype(bool)
                samples_from_class = features[from_class]
                labels_from_class = labels[from_class]

                tf_dataset_from_class = tf.data.Dataset.from_tensor_slices(
                    (samples_from_class, labels_from_class))
                tf_dataset_from_class = tf_dataset_from_class \
                    .repeat().shuffle(100, reshuffle_each_iteration=True)
                datasets_from_class.append(tf_dataset_from_class)
            dataset = tf.data.Dataset.sample_from_datasets(
                datasets_from_class)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (features, labels['astro_class']))

        dataset = dataset.batch(self.batch_size).prefetch(5)
        return dataset

    def _label_list_to_one_hot(self, label_list: np.ndarray):
        matches = np.stack(
            [label_list == s for s in self.list_of_classes], axis=-1)
        onehot_labels = matches.astype(np.float32)
        return onehot_labels

    def _evaluate_training_stopper(self, current_validation_loss: float) -> bool:
        if current_validation_loss < self.best_historic_loss:
            self.best_historic_loss = current_validation_loss
            self.validations_without_improvement = 0
            return False

        self.validations_without_improvement += 1
        if self.validations_without_improvement >= self.max_validations_without_improvement:
            return True
        else:
            return False
