# src/models/Resnet_model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import resnet_v2
from typing import Dict, List

class ResNetStampModel(tf.keras.Model):
    def __init__(
        self,
        dense_config: List[Dict],
        dropout_rate: float,
        use_batchnorm_metadata: bool,
        num_classes: int,
        use_metadata: bool,
        input_shape=(63, 63, 3), # Tamaño típico o esperado
        resize_target=(224, 224), # ResNet funciona mejor con imágenes más grandes
        fine_tune_at: int = 0, # 0 para congelar todo, >0 para descongelar capas superiores
        weights: str = "imagenet",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_config = dense_config
        self.dropout_rate = dropout_rate
        self.use_batchnorm_metadata = use_batchnorm_metadata
        self.num_classes = num_classes
        self.use_metadata = use_metadata

        if resize_target is not None:
            self.resize_target = tuple(resize_target)
        else:
            self.resize_target = None
        # ---------------------------------------------------------

        # 1. Backbone: ResNet50V2
        # Ahora self.resize_target es una tupla, así que (H, W) + (3,) funciona.
        shape_for_backbone = self.resize_target + (3,) if self.resize_target else input_shape
        
        # 1. Backbone: ResNet50V2
        # include_top=False devuelve los features antes de la clasificación
        self.backbone = resnet_v2.ResNet50V2(
            include_top=False,
            weights=weights,
            input_shape=shape_for_backbone,
            pooling='avg' # Devuelve vector (Batch, 2048)
        )
        
        # Configuración de congelamiento (Freezing)
        if fine_tune_at == 0:
            self.backbone.trainable = False
        else:
            self.backbone.trainable = True
            # Congelar todas las capas antes de fine_tune_at
            for layer in self.backbone.layers[:-fine_tune_at]:
                layer.trainable = False

        self.dropout = Dropout(self.dropout_rate, name="dropout")

        # 2. Capas densas (Head)
        self.dense_layers = []
        for i, cfg in enumerate(dense_config):
            self.dense_layers.append(
                Dense(
                    units=cfg['units'],
                    activation=cfg.get('activation', 'relu'),
                    name=f"dense_head_{i+1}"
                )
            )

        # 3. Metadata
        self.metadata_batchnorm = BatchNormalization(name="bn_metadata") if self.use_batchnorm_metadata else None

        # 4. Salida
        self.output_layer = Dense(self.num_classes, name="logits", activation='softmax')

    def _preprocess_image(self, img):
        """Ajusta la imagen para ResNet"""
        # 1. Asegurar 3 canales (si viene en 1 canal, repetir; si viene en más, cortar)
        if img.shape[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)
        elif img.shape[-1] > 3:
            img = img[:, :, :, :3] # Tomar solo los primeros 3 (g,r,i usualmente)
            
        # 2. Resize (ResNet pre-entrenada prefiere 224x224, aunque acepta otros tamaños)
        if self.resize_target is not None:
            img = tf.image.resize(img, self.resize_target)
            
        # 3. Preprocesamiento específico de ResNetV2 (escala de -1 a 1)
        return resnet_v2.preprocess_input(img)

    def call(self, inputs, training=False):
        x_img, x_metadata = inputs
        x_img = x_img[..., :3] 
        # --- Lógica de Invarianza Rotacional (Igual a CNN_model) ---
        # Generar 8 vistas (4 rotaciones + 4 flips)
        rot_list = [x_img, 
                    tf.image.rot90(x_img, k=1), 
                    tf.image.rot90(x_img, k=2), 
                    tf.image.rot90(x_img, k=3)]
        
        flipped_list = [tf.image.flip_up_down(r) for r in rot_list]
        full_views = rot_list + flipped_list # Total 8 imágenes por sample
        
        # Procesar cada vista por el Backbone
        view_features = []
        for view in full_views:
            view = self._preprocess_image(view)
            feat = self.backbone(view, training=training) # (Batch, 2048)
            view_features.append(feat)
            
        # Promediar las características de las 8 vistas
        x = tf.stack(view_features, axis=0) # (8, Batch, 2048)
        x = tf.reduce_mean(x, axis=0)       # (Batch, 2048)
        
        x = self.dropout(x, training=training)

        # --- Metadata ---
        if self.use_batchnorm_metadata:
            x_metadata = self.metadata_batchnorm(x_metadata, training=training)

        if self.use_metadata:
            x = tf.concat([x, x_metadata], axis=-1)

        # --- Dense Head ---
        for layer in self.dense_layers:
            x = layer(x)

        return self.output_layer(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "dense_config": self.dense_config,
            "dropout_rate": self.dropout_rate,
            "use_batchnorm_metadata": self.use_batchnorm_metadata,
            "num_classes": self.num_classes,
            "use_metadata": self.use_metadata,
            "resize_target": self.resize_target
        })
        return config

    @classmethod
    def from_config(cls, config: Dict):
        return cls(**config)