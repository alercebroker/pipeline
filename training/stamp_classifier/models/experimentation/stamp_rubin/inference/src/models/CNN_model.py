import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from typing import List, Dict


# Asegúrate de tener esta función si usas cropping
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
    return im_with_mask

class DynamicStampModel(tf.keras.Model):
    """
    A flexible CNN + MLP model constructed from configuration lists.
    """
    def __init__(
        self,
        conv_config,
        dense_config,
        dropout_rate,
        use_batchnorm_metadata,  # nombre más explícito
        num_classes,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_config = conv_config
        self.dense_config = dense_config
        self.dropout_rate = dropout_rate
        self.use_batchnorm_metadata = use_batchnorm_metadata
        self.num_classes = num_classes

        # Capas convolucionales (sin batchnorm)
        self.conv_layers = []
        for i, cfg in enumerate(conv_config):
            self.conv_layers.append(
                Conv2D(
                    filters=cfg['filters'],
                    kernel_size=cfg.get('kernel_size', (3, 3)),
                    activation=cfg.get('activation', 'relu'),
                    #strides = (2,2),
                    padding='same',
                    name=f"conv_{i+1}"
                )
            )
            if cfg.get('pool', False):
                self.conv_layers.append(
                    MaxPooling2D(pool_size=cfg.get('pool_size', (2, 2)), name=f"pool_{i+1}")
                )

        self.flatten = Flatten(name="flatten")
        self.dropout = Dropout(self.dropout_rate, name="dropout")

        # Capas densas (sin batchnorm)
        self.dense_layers = []
        for i, cfg in enumerate(dense_config):
            self.dense_layers.append(
                Dense(
                    units=cfg['units'],
                    activation=cfg.get('activation', 'relu'),
                    name=f"dense_{i+1}"
                )
            )
        self.dense1 = Dense(units=32,activation='relu')

        # Batchnorm para la metadata (solo si se activa)
        self.metadata_batchnorm = BatchNormalization(name="bn_metadata") if self.use_batchnorm_metadata else None

        # Capa final
        self.output_layer = Dense(self.num_classes, name="logits")

        
    def call(self, inputs, training=False):
        x_img, x_metadata = inputs
        #x_img = add_center_mask(x_img)  # Añadir máscara central si es necesario

        rot_list = [x_img, tf.image.rot90(x_img), 
                    tf.image.rot90(x_img, k=2), tf.image.rot90(x_img, k=3)]

        for i in range(4):
            rot_list.append(tf.image.flip_up_down(rot_list[i]))
        rot_list = [x_img]

        output_list = []
        for im in rot_list:
        # Pipeline convolucional
            for layer in self.conv_layers:
                im = layer(im)

            x = self.flatten(im)
            output_list.append(x)
        
        x = tf.stack(output_list, axis=0)
        x = tf.reduce_mean(x, axis=0)
        x = self.dropout(x, training=training)

        # Normalización de metadata (si aplica)
        if self.use_batchnorm_metadata:
            x_metadata = self.metadata_batchnorm(x_metadata, training=training)

        # Combinar imagen + metadata
        #x = tf.concat([x, x_metadata], axis=-1)

        # Pipeline dense
        #print(x.shape)
        for layer in self.dense_layers:
            x = layer(x)

        return self.output_layer(x)
    
    def get_config(self) -> Dict:
        """ Devuelve la configuración del modelo para su serialización """
        config = super().get_config()
        config.update({
            "conv_config": self.conv_config,
            "dense_config": self.dense_config,
            "dropout_rate": self.dropout_rate,
            "use_batchnorm_metadata": self.use_batchnorm_metadata,
            "num_classes": self.num_classes,
            })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """ Permite reconstruir el modelo desde la configuración """
        return cls(**config)