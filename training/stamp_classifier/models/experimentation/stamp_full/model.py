import tensorflow as tf
import pandas as pd

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model
from typing import List, Dict
from sklearn.preprocessing import QuantileTransformer

def add_center_mask(im, center_size=8):
    """
    im: tensor [batch, 63, 63, 6]
    center_size: tamaño del cuadrado central (8 por defecto)
    returns: tensor [batch, 63, 63, 7]
    """
    batch_size, h, w, _ = im.shape

    # Coordenadas para el centro
    start_h = (h - center_size) // 2
    end_h = start_h + center_size
    start_w = (w - center_size) // 2
    end_w = start_w + center_size

    # Crear máscara [63, 63] con ceros
    mask = tf.zeros((h, w), dtype=tf.float32)

    # Poner 1s en la región central
    ones_center = tf.ones((center_size, center_size), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.reshape(tf.stack(tf.meshgrid(
            tf.range(start_h, end_h),
            tf.range(start_w, end_w),
            indexing="ij"
        ), axis=-1), [-1, 2]),
        updates=tf.reshape(ones_center, [-1])
    )

    # Expandir y replicar la máscara para el batch
    mask = tf.expand_dims(mask, axis=0)  # [1, 63, 63]
    mask = tf.tile(mask, [batch_size, 1, 1])  # [batch, 63, 63]
    mask = tf.expand_dims(mask, axis=-1)  # [batch, 63, 63, 1]

    # Concatenar como nuevo canal
    im_with_mask = tf.concat([im, mask], axis=-1)  # [batch, 63, 63, 7]
    im_with_mask = im[:,start_h:end_h, start_w:end_w, :]  # [batch, 8, 8, 6]
    return im_with_mask

import tensorflow as tf
import math

def positional_encoding_ra_dec(ra_deg, dec_deg, embedding_size=32):
    """
    Positional encoding sinusoidal con dimensión fija para coordenadas RA/DEC.

    Args:
        ra_deg: tensor [batch] o [batch, 1] con RA en grados (0 a 360)
        dec_deg: tensor [batch] o [batch, 1] con DEC en grados (-90 a +90)
        embedding_size: dimensión total del encoding (debe ser par)

    Returns:
        pe: tensor [batch, embedding_size]
    """
    assert embedding_size % 2 == 0, "embedding_size debe ser par"

    half_dim = embedding_size // 2
    ra_dim = half_dim // 2
    dec_dim = half_dim - ra_dim  # en caso de número impar

    # Convertir a radianes
    ra_rad = tf.expand_dims(ra_deg * math.pi / 180.0, -1)  # [batch, 1]
    dec_rad = tf.expand_dims(dec_deg * math.pi / 180.0, -1)

    # Frecuencias tipo transformer
    ra_freqs = tf.pow(10000.0, tf.range(0, ra_dim, dtype=tf.float32) / ra_dim)  # [ra_dim]
    dec_freqs = tf.pow(10000.0, tf.range(0, dec_dim, dtype=tf.float32) / dec_dim)

    ra_scaled = ra_rad / ra_freqs  # [batch, ra_dim]
    dec_scaled = dec_rad / dec_freqs

    ra_pe = tf.concat([tf.sin(ra_scaled), tf.cos(ra_scaled)], axis=-1)  # [batch, 2 * ra_dim]
    dec_pe = tf.concat([tf.sin(dec_scaled), tf.cos(dec_scaled)], axis=-1)  # [batch, 2 * dec_dim]

    pe = tf.concat([ra_pe, dec_pe], axis=-1)  # [batch, embedding_size]

    return pe



class StampModelModified(Model):
    def __init__(
            self,
            layer_sizes: List[int], 
            dropout_rate: float,
            with_batchnorm: bool, 
            first_kernel_size: int, 
            dict_mapping_classes: int,
            with_crop: bool,
            order_features: List[str] = None,
            norm_means: List[float] = None,
            norm_stds: List[float] = None,
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
        self.with_crop = with_crop
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
        
        self.conv_5 = Conv2D(
            layer_sizes[4], (3, 3),
            activation='relu', padding='same',
            kernel_initializer=tf.keras.initializers.HeUniform())


        self.with_batchnorm = with_batchnorm
        if self.with_batchnorm:
            self.bn = BatchNormalization()

        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.dense_1 = Dense(layer_sizes[5], activation='tanh')
        self.dense_2 = Dense(n_classes)

    def call(self, inputs, training=False):
        x, pos = inputs[0], inputs[1]
        rot_list = [x, tf.image.rot90(x), tf.image.rot90(x, k=2), tf.image.rot90(x, k=3)]

        for i in range(4):
            rot_list.append(tf.image.flip_up_down(rot_list[i]))

        output_list = []
        for im in rot_list:
            if self.with_crop:
                im = add_center_mask(im, center_size=23)
            x = self.conv_1(im)
            x = self.conv_2(x)
            x = self.pool_1(x)
            x = self.conv_3(x)
            x = self.conv_4(x)
            x = self.conv_5(x)
            x = self.pool_2(x)

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
            "norm_stds": self.norm_stds,
            "with_crop": self.with_crop
            })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """ Permite reconstruir el modelo desde la configuración """
        return cls(**config)
    


class StampModelFull(Model):
    def __init__(
            self,
            layer_sizes: List[int], 
            dropout_rate: float,
            with_batchnorm: bool, 
            first_kernel_size: int, 
            dict_mapping_classes: int,
            with_crop: bool,
            **kwargs):
        super().__init__(**kwargs)

        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.with_batchnorm = with_batchnorm
        self.first_kernel_size = first_kernel_size
        self.dict_mapping_classes = dict_mapping_classes
        self.with_crop = with_crop
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
            if self.with_crop:
                im = add_center_mask(im, center_size=23)
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
            "with_crop": self.with_crop,
            })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """ Permite reconstruir el modelo desde la configuración """
        return cls(**config)