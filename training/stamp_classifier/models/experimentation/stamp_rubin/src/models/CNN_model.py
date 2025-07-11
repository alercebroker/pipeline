import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


# Asegúrate de tener esta función si usas cropping
def add_center_mask(x, center_size=23):
    # Placeholder: implementa según tu lógica
    return x

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
        self.use_batchnorm_metadata = use_batchnorm_metadata
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Capas convolucionales (sin batchnorm)
        self.conv_layers = []
        for i, cfg in enumerate(conv_config):
            self.conv_layers.append(
                Conv2D(
                    filters=cfg['filters'],
                    kernel_size=cfg.get('kernel_size', (3, 3)),
                    activation=cfg.get('activation', 'relu'),
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

        # Batchnorm para la metadata (solo si se activa)
        self.metadata_batchnorm = BatchNormalization(name="bn_metadata") if self.use_batchnorm_metadata else None

        # Capa final
        self.output_layer = Dense(self.num_classes, name="logits")

        
    def call(self, inputs, training=False):
        x_img, x_metadata = inputs

        # Pipeline convolucional
        for layer in self.conv_layers:
            x_img = layer(x_img)

        x = self.flatten(x_img)
        x = self.dropout(x, training=training)

        # Normalización de metadata (si aplica)
        if self.use_batchnorm_metadata:
            x_metadata = self.metadata_batchnorm(x_metadata, training=training)

        # Combinar imagen + metadata
        x = tf.concat([x, x_metadata], axis=-1)

        # Pipeline dense
        for layer in self.dense_layers:
            x = layer(x)

        return self.output_layer(x)