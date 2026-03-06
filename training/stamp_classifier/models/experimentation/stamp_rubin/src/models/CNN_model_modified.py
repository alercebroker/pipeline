import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D, Concatenate
from typing import List, Dict

class MultiResStampModel(tf.keras.Model):
    """
    Modelo Multiresolución Híbrido.
    Soporta entradas flexibles:
    1. Diccionario (MultiRes): {'main': tensor, 'level0': tensor, 'metadata': tensor...}
    2. Tupla (Legacy): (stamps_tensor, metadata_tensor)
    """
    def __init__(
        self,
        conv_config: List[Dict],
        dense_config: List[Dict],
        dropout_rate: float,
        num_classes: int,
        use_metadata: bool = False,
        use_batchnorm_metadata: bool = False,
        input_keys: List[str] = ['main'], # Lista de claves de imágenes esperadas
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_config = conv_config
        self.dense_config = dense_config
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        self.use_batchnorm_metadata = use_batchnorm_metadata
        self.input_keys = input_keys

        # --- Ramas Convolucionales ---
        # Creamos una rama INDEPENDIENTE para cada input key de IMAGEN.
        # Si estamos en modo legacy, 'main' será la única clave.
        self.branches = {}
        
        for key in self.input_keys:
            layers = []
            for i, cfg in enumerate(conv_config):
                layers.append(
                    Conv2D(
                        filters=cfg['filters'],
                        kernel_size=cfg.get('kernel_size', (3, 3)),
                        activation=cfg.get('activation', 'relu'),
                        padding='same',
                        name=f"branch_{key}_conv_{i+1}"
                    )
                )
                if cfg.get('pool', False):
                    layers.append(
                        MaxPooling2D(pool_size=cfg.get('pool_size', (2, 2)), name=f"branch_{key}_pool_{i+1}")
                    )
            
            # Capas finales de la rama (Global Pooling)
            layers.append(BatchNormalization(name=f"branch_{key}_bn"))
            layers.append(GlobalAveragePooling2D(name=f"branch_{key}_gap"))
            layers.append(Dropout(self.dropout_rate, name=f"branch_{key}_dropout"))
            
            # Registrar layers en el modelo
            branch_name = f"layers_{key}"
            setattr(self, branch_name, layers)
            self.branches[key] = layers

        # --- Rama Metadatos (Opcional) ---
        self.metadata_bn = BatchNormalization(name="bn_metadata") if self.use_batchnorm_metadata else None

        # --- Capas Densas (Fusion) ---
        self.dense_layers = []
        for i, cfg in enumerate(dense_config):
            self.dense_layers.append(
                Dense(
                    units=cfg['units'],
                    activation=cfg.get('activation', 'relu'),
                    name=f"dense_fusion_{i+1}"
                )
            )

        self.output_layer = Dense(self.num_classes, activation='linear', name="logits")

    def call(self, inputs, training=False):
        # 1. Normalizar entrada a formato Diccionario
        x_dict = {}
        x_metadata = None

        if isinstance(inputs, dict):
            # Caso MultiRes o Legacy Dictionary
            x_dict = inputs
            if 'metadata' in inputs:
                x_metadata = inputs['metadata']
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            # Caso Legacy Tuple: (stamps, metadata)
            # Asumimos que la imagen principal se mapea a la primera clave de input_keys (usualmente 'main')
            main_key = self.input_keys[0]
            x_dict[main_key] = inputs[0]
            x_metadata = inputs[1]
        else:
            raise ValueError(f"Formato de entrada no soportado: {type(inputs)}")

        # 2. Procesar Ramas de Imágenes
        branch_outputs = []

        for key in self.input_keys:
            if key not in x_dict:
                # Si una clave esperada no viene, saltamos (o podríamos lanzar error)
                continue
                
            x_img = x_dict[key] # (Batch, H, W, C)

            # --- Augmentación (Rotations + Flips) ---
            # Aplicamos la misma lógica de 8 variaciones
            rot0 = x_img
            rot90 = tf.image.rot90(x_img, k=1)
            rot180 = tf.image.rot90(x_img, k=2)
            rot270 = tf.image.rot90(x_img, k=3)
            
            variations = [
                rot0, tf.image.flip_up_down(rot0),
                rot90, tf.image.flip_up_down(rot90),
                rot180, tf.image.flip_up_down(rot180),
                rot270, tf.image.flip_up_down(rot270)
            ]
            x_concat = tf.concat(variations, axis=0) # [8*Batch, H, W, C]

            # --- Pasar por la rama convolucional correspondiente ---
            layers = getattr(self, f"layers_{key}")
            
            feat = x_concat
            for layer in layers:
                feat = layer(feat, training=training)
            
            # feat shape: [8*Batch, Features]
            
            # --- Average Stack ---
            feature_dim = tf.shape(feat)[-1]
            batch_size = tf.shape(x_img)[0]
            
            feat = tf.reshape(feat, (8, batch_size, feature_dim))
            feat = tf.reduce_mean(feat, axis=0) # [Batch, Features]
            
            branch_outputs.append(feat)

        # 3. Concatenación de Ramas Visuales
        if len(branch_outputs) > 1:
            x_fusion = Concatenate(axis=-1)(branch_outputs)
        elif len(branch_outputs) == 1:
            x_fusion = branch_outputs[0]
        else:
            raise ValueError("No se procesó ninguna rama de imagen válida.")

        # 4. Inyección de Metadatos (Opcional)
        if self.use_metadata and x_metadata is not None:
            if self.use_batchnorm_metadata:
                x_metadata = self.metadata_bn(x_metadata, training=training)
            x_fusion = Concatenate(axis=-1)([x_fusion, x_metadata])

        # 5. Capas Densas Finales
        for layer in self.dense_layers:
            x_fusion = layer(x_fusion)

        return self.output_layer(x_fusion)

    def get_config(self):
        config = super().get_config()
        config.update({
            "conv_config": self.conv_config,
            "dense_config": self.dense_config,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
            "use_metadata": self.use_metadata,
            "use_batchnorm_metadata": self.use_batchnorm_metadata,
            "input_keys": self.input_keys
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DynamicStampModelModified(tf.keras.Model):
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
        use_metadata,
        architecture_mode='classic',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_config = conv_config
        self.dense_config = dense_config
        self.dropout_rate = dropout_rate
        self.use_batchnorm_metadata = use_batchnorm_metadata
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        self.architecture_mode = architecture_mode

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

        self.bn_features = BatchNormalization(name="bn_features_stacked")
        self.global_pool = GlobalAveragePooling2D(name="adaptive_avg_pool_1x1")
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
        self.output_layer = Dense(self.num_classes, name="logits",activation='linear')


    def call(self, inputs, training=False):
        x_img, x_metadata = inputs

        # --- Generación de las 8 Variantes ---
        # Creamos una lista de tensores
        rot0 = x_img
        rot90 = tf.image.rot90(x_img, k=1)
        rot180 = tf.image.rot90(x_img, k=2)
        rot270 = tf.image.rot90(x_img, k=3)
        
        # Lista con las 4 rotaciones y sus respectivos flips
        variations = [
            rot0, tf.image.flip_up_down(rot0),
            rot90, tf.image.flip_up_down(rot90),
            rot180, tf.image.flip_up_down(rot180),
            rot270, tf.image.flip_up_down(rot270)
        ] # Lista de 8 tensores de shape [Batch, H, W, 3]

        # Concatenamos en el eje del Batch para procesar todo junto (Más eficiente que un for loop)
        # Shape: [8 * Batch, H, W, 3]
        x_concat = tf.concat(variations, axis=0)

        # --- Pipeline Convolucional ---
        x = x_concat
        for layer in self.conv_layers:
            x = layer(x)
        # Salida x: [8 * Batch, H_out, W_out, Filters]

        # --- Average Stack (Punto clave de la modificación) ---
        # Recuperamos la dimensión de las variantes para promediar
        # Reshape a [8, Batch, H_out, W_out, Filters]
        feature_shape = tf.shape(x)[1:] # [H_out, W_out, Filters]
        batch_size = tf.shape(x_img)[0]
        
        x = tf.reshape(x, (8, batch_size, feature_shape[0], feature_shape[1], feature_shape[2]))
        x = tf.reduce_mean(x, axis=0)

        x = self.bn_features(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)

        # Normalización de metadata (si aplica)
        if self.use_batchnorm_metadata:
            x_metadata = self.metadata_batchnorm(x_metadata, training=training)

        # Combinar imagen + metadata
        if self.use_metadata:
            x = tf.concat([x, x_metadata], axis=-1)

        # Pipeline dense
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
            "use_metadata":self.use_metadata,
            "architecture_mode":self.architecture_mode
            })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """ Permite reconstruir el modelo desde la configuración """
        return cls(**config)