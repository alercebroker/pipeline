# src/models/ResNetCustom.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Model

class ResidualBlock(tf.keras.layers.Layer):
    """
    Bloque Residual Básico:
    Input -> [Conv-BN-ReLU] -> [Conv-BN] -> (+) -> ReLU
                                     |       ^
                                     |_______|
    """
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.filters = filters
        
        # Capa 1: Puede hacer downsampling si stride > 1
        self.conv1 = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        
        # Capa 2: Mantiene dimensiones
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        
        # Shortcut (Atajo):
        # Si cambiamos el tamaño (stride > 1) o el número de filtros, 
        # necesitamos adaptar la entrada para poder sumarla.
        self.shortcut_conv = None
        if stride != 1:
            self.shortcut_conv = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)
            self.shortcut_bn = BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Lógica del shortcut
        shortcut = inputs
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        elif inputs.shape[-1] != self.filters:
            # Caso raro: stride 1 pero aumentan canales (transición de bloque)
            # En esta implementación forzamos la proyección si los canales no coinciden
            # aunque sea stride 1.
             shortcut = Conv2D(self.filters, 1, 1, padding='same', use_bias=False)(shortcut)
             shortcut = BatchNormalization()(shortcut, training=training)

        x = Add()([x, shortcut])
        return self.relu(x)

class ResNetCustomMini(Model):
    """
    Mini-ResNet basada en la tesis (Sulima-Lyczkowski, 2025).
    Filtros: 16 -> 32 -> 64.
    Parámetros aprox: ~0.5 Millones.
    """
    def __init__(self, num_classes, dropout_rate=0.5, use_metadata=False, use_batchnorm_metadata=True, **kwargs):
        super().__init__(**kwargs)
        self.use_metadata = use_metadata
        self.dropout_rate = dropout_rate
        self.use_batchnorm_metadata = use_batchnorm_metadata

        # --- ARQUITECTURA ---
        
        # 1. Stem (Tallo inicial)
        # Recibe (31, 31, 6). Sale (31, 31, 16)
        self.conv1 = Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()

        # 2. Etapa 1 (16 Filtros) - Mantener tamaño 31x31
        # Usamos 3 bloques (aprox 6 capas)
        self.layer1 = tf.keras.Sequential([
            ResidualBlock(16, stride=1),
            ResidualBlock(16, stride=1),
            ResidualBlock(16, stride=1)
        ])

        # 3. Etapa 2 (32 Filtros) - Downsample a 16x16
        # Usamos 3 bloques
        self.layer2 = tf.keras.Sequential([
            ResidualBlock(32, stride=2), # Aquí baja a la mitad
            ResidualBlock(32, stride=1),
            ResidualBlock(32, stride=1)
        ])

        # 4. Etapa 3 (64 Filtros) - Downsample a 8x8
        # Usamos 3 bloques
        self.layer3 = tf.keras.Sequential([
            ResidualBlock(64, stride=2), # Aquí baja a la mitad
            ResidualBlock(64, stride=1),
            ResidualBlock(64, stride=1)
        ])

        # 5. Head (Clasificador)
        self.global_pool = GlobalAveragePooling2D()
        self.bn_final = BatchNormalization()
        
        # Batchnorm opcional para metadata
        if self.use_batchnorm_metadata:
            self.bn_metadata = BatchNormalization()

        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_classes, activation='softmax', dtype='float32')

    def call(self, inputs, training=False):
        x_img, x_metadata = inputs
        # x_img shape: (Batch, 31, 31, 6)

        # --- Data Augmentation (TTA interno) ---
        rot_list = [x_img, 
                    tf.image.rot90(x_img, k=1), 
                    tf.image.rot90(x_img, k=2), 
                    tf.image.rot90(x_img, k=3)]
        flipped_list = [tf.image.flip_up_down(r) for r in rot_list]
        full_views = rot_list + flipped_list

        view_features = []
        for view in full_views:
            # Stem
            x = self.conv1(view)
            x = self.bn1(x, training=training)
            x = self.relu(x)

            # Bloques Residuales
            x = self.layer1(x, training=training)
            x = self.layer2(x, training=training)
            x = self.layer3(x, training=training)

            # Pooling (Vector de 64 features)
            x = self.global_pool(x)
            view_features.append(x)

        # Promedio de vistas
        x = tf.stack(view_features, axis=0)
        x = tf.reduce_mean(x, axis=0)

        # Procesamiento final
        x = self.bn_final(x, training=training)

        if self.use_metadata:
            if self.use_batchnorm_metadata:
                x_metadata = self.bn_metadata(x_metadata, training=training)
            x = tf.concat([x, x_metadata], axis=-1)

        x = self.dropout(x, training=training)
        return self.classifier(x)