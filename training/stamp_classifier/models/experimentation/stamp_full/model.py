import tensorflow as tf
import pandas as pd

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model
from typing import List, Dict


class StampModelFull(Model):
    def __init__(
            self,
            layer_sizes: List[int], 
            dropout_rate: float,
            with_batchnorm: bool, 
            first_kernel_size: int, 
            dict_mapping_classes: int,
            order_features: List[str] = None,
            norm_means: pd.Series = None,
            norm_stds: pd.Series = None,
            **kwargs):
        super().__init__(**kwargs)

        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.with_batchnorm = with_batchnorm
        self.first_kernel_size = first_kernel_size
        self.dict_mapping_classes = dict_mapping_classes
        self.order_features = order_features if order_features is not None else []
        self.norm_means = norm_means
        self.norm_stds = norm_stds
        n_classes = len(dict_mapping_classes)

        self.conv_1 = Conv2D(
            layer_sizes[0], (first_kernel_size, first_kernel_size),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.HeUniform())
        self.pool_1 = MaxPooling2D((2, 2), padding='same')
        self.conv_2 = Conv2D(
            layer_sizes[1], (3, 3),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.HeUniform())
        self.pool_2 = MaxPooling2D((2, 2))
        self.conv_3 = Conv2D(
            layer_sizes[2], (3, 3),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.HeUniform())
        self.pool_3 = MaxPooling2D((2, 2))
        self.conv_4 = Conv2D(
            layer_sizes[3], (3, 3),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.HeUniform())

        self.with_batchnorm = with_batchnorm
        if self.with_batchnorm:
            self.bn = BatchNormalization()

        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.dense_1 = Dense(layer_sizes[4], activation='tanh')
        self.dense_2 = Dense(n_classes)

    def call(self, inputs, training=False):
        x, pos = inputs[0], inputs[1]
        rot_list = [x, tf.image.rot90(x), tf.image.rot90(x, k=2), tf.image.rot90(x, k=3)]

        for i in range(4):
            rot_list.append(tf.image.flip_up_down(rot_list[i]))

        output_list = []
        for im in rot_list:
            x = self.conv_1(im)
            x = self.pool_1(x)

            x = self.conv_2(x)
            x = self.pool_2(x)

            x = self.conv_3(x)
            x = self.pool_3(x)

            x = self.conv_4(x)

            x = self.flatten(x)
            output_list.append(x)

        x = tf.stack(output_list, axis=0)
        x = tf.reduce_mean(x, axis=0)

        x = self.dropout(x, training=training)

        if self.with_batchnorm:
            pos = self.bn(pos, training=training)

        x = tf.concat([x, pos], axis=1)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x

    def get_config(self) -> Dict:
        """ Devuelve la configuración del modelo para su serialización """
        config = super().get_config()
        config.update({
            "layer_sizes": self.layer_sizes,
            "dropout_rate": self.dropout_rate,
            "with_batchnorm": self.with_batchnorm,
            "first_kernel_size": self.first_kernel_size,
            "dict_mapping_classes": self.dict_mapping_classes,
            "order_features": self.order_features,
            "norm_means": self.norm_means,
            "norm_stds": self.norm_stds
        })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """ Permite reconstruir el modelo desde la configuración """
        return cls(**config)