# src/models/ResNetCustom.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras import Model

class ResidualBlock(tf.keras.layers.Layer):
    """
    Bloque Residual Básico (BasicBlock) como se describe en He et al. 2016
    y utilizado en la tesis.
    Estructura: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        
        # Primera convolución (puede reducir dimensionalidad si stride > 1)
        self.conv1 = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        
        # Segunda convolución
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        
        # Shortcut (conexión de atajo)
        # Si las dimensiones cambian (stride > 1 o cambio de filtros), 
        # necesitamos proyectar la entrada para que coincida.
        self.shortcut_conv = None
        if stride != 1:
            self.shortcut_conv = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)
            self.shortcut_bn = BatchNormalization()

    def call(self, inputs, training=False):
        # Rama principal
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Rama shortcut
        shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        elif inputs.shape[-1] != self.filters:
            # Caso especial: si stride=1 pero aumentan los filtros (transición entre bloques)
            # necesitamos proyección 1x1
            # Nota: En la tesis esto se maneja en la definición del bloque, aquí lo generalizamos.
             # Sin embargo, en esta implementación, el stride maneja el cambio de filtros.
             # Si hubiera discrepancia, se añade aquí.
             pass

        # Suma y activación final
        x = Add()([x, shortcut])
        x = self.relu(x)
        return x

class ResNetCustom(Model):
    """
    Implementación de la ResNet de la tesis (Sulima-Lyczkowski, 2025).
    Adaptada para estampillas de 31x31 con 6 canales.
    
    Arquitectura basada en Tablas 5.3 y 5.4:
    - Entrada: 6 canales
    - Bloque 16: 11 capas (1 Conv inicial + 5 Bloques Residuales)
    - Bloque 32: 10 capas (5 Bloques Residuales)
    - Bloque 64: 10 capas (5 Bloques Residuales)
    - Total: 31 capas convolucionales.
    """
    def __init__(self, num_classes, dropout_rate=0.5, use_metadata=False, use_batchnorm_metadata=True, **kwargs):
        super().__init__(**kwargs)
        self.use_metadata = use_metadata
        self.dropout_rate = dropout_rate
        self.use_batchnorm_metadata = use_batchnorm_metadata

        # --- CONSTRUCCIÓN DE LA RED (Feature Extractor) ---
        
        # BLOQUE 16 (Output: 16 canales)
        # Según tesis: 11 capas. Asumimos: 1 Conv inicial + 5 Bloques Residuales (2 capas c/u)
        self.block16 = tf.keras.Sequential(name="block16")
        # Entrada directa de 6 canales
        self.block16.add(Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, input_shape=(None, None, 6)))
        self.block16.add(BatchNormalization())
        self.block16.add(ReLU())
        for _ in range(5):
            self.block16.add(ResidualBlock(filters=16, stride=1))

        # BLOQUE 32 (Output: 32 canales)
        # Según tesis: 10 capas = 5 Bloques Residuales.
        # El primero hace downsampling (stride 2) para reducir espacialmente (31x31 -> 16x16)
        self.block32 = tf.keras.Sequential(name="block32")
        self.block32.add(ResidualBlock(filters=32, stride=2)) # Downsample
        for _ in range(4):
            self.block32.add(ResidualBlock(filters=32, stride=1))

        # BLOQUE 64 (Output: 64 canales)
        # Según tesis: 10 capas = 5 Bloques Residuales.
        # El primero hace downsampling (stride 2) (16x16 -> 8x8)
        self.block64 = tf.keras.Sequential(name="block64")
        self.block64.add(ResidualBlock(filters=64, stride=2)) # Downsample
        for _ in range(4):
            self.block64.add(ResidualBlock(filters=64, stride=1))

        # Pooling final (Adaptive Avg Pool -> Global Avg Pool)
        # Reduce (Batch, 8, 8, 64) -> (Batch, 64)
        self.global_pool = GlobalAveragePooling2D()

        # --- HEAD CLASIFICADOR ---
        # Tesis Tabla 5.4 Etapa 6: BatchNorm2d + Flatten -> Linear
        self.bn_final = BatchNormalization(name="bn_final")
        self.flatten = Flatten()
        
        if self.use_batchnorm_metadata:
            self.bn_metadata = BatchNormalization(name="bn_metadata")

        self.dropout = Dropout(dropout_rate)
        
        # La tesis usa una sola capa lineal al final, pero tú sueles usar una configuración
        # 'dense_config'. Aquí sigo la tesis (una capa de salida), pero si quieres capas ocultas
        # previas, puedes agregarlas.
        # Tesis: Linear(64 -> 5)
        self.classifier = Dense(num_classes, activation='softmax', name="logits")

    def call(self, inputs, training=False):
        x_img, x_metadata = inputs
        # x_img shape: (Batch, 31, 31, 6)

        # --- 1. ROTACIONES Y FLIPS (Tesis Tabla 5.4 Etapa 1) ---
        # Generar 8 variantes
        rot_list = [x_img, 
                    tf.image.rot90(x_img, k=1), 
                    tf.image.rot90(x_img, k=2), 
                    tf.image.rot90(x_img, k=3)]
        
        flipped_list = [tf.image.flip_up_down(r) for r in rot_list]
        full_views = rot_list + flipped_list # Total 8 tensores

        # --- 2. PROCESAMIENTO (Shared Weights) ---
        view_features = []
        for view in full_views:
            # Pasa por los bloques residuales
            x = self.block16(view, training=training)
            x = self.block32(x, training=training)
            x = self.block64(x, training=training)
            
            # Pooling (Reduce a vector 1x1x64 -> 64)
            x = self.global_pool(x)
            view_features.append(x)

        # --- 3. STACK + MEAN (Tesis Tabla 5.4 Etapa 5) ---
        # Promediar los vectores de características de las 8 vistas
        x = tf.stack(view_features, axis=0) # (8, Batch, 64)
        x = tf.reduce_mean(x, axis=0)       # (Batch, 64)

        # --- 4. MLP FINAL (Tesis Tabla 5.4 Etapa 6) ---
        x = self.bn_final(x, training=training)
        
        # Metadata (Opcional según tu pipeline, la tesis bimodal lo usa)
        if self.use_metadata:
            if self.use_batchnorm_metadata:
                x_metadata = self.bn_metadata(x_metadata, training=training)
            x = tf.concat([x, x_metadata], axis=-1)

        x = self.dropout(x, training=training)
        
        # Clasificación
        return self.classifier(x)

    def get_config(self):
        return {
            "num_classes": self.classifier.units,
            "dropout_rate": self.dropout_rate,
            "use_metadata": self.use_metadata,
            "use_batchnorm_metadata": self.use_batchnorm_metadata
        }