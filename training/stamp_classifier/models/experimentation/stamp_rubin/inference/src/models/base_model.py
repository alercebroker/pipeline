import mlflow
import tensorflow as tf

#from sklearn.metrics import precision_recall_fscore_support

from src.training.losses import balanced_xentropy

class CustomModel(tf.keras.Model):
    def __init__(self, stamp_classifier, loss_object, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.stamp_classifier = stamp_classifier
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")
        self.val_precision = tf.keras.metrics.Precision(name="val_precision")
        self.val_recall = tf.keras.metrics.Recall(name="val_recall")

    def train_step(self, data):
        (stamps, metadata), labels = data
        with tf.GradientTape() as tape:
            predictions = self.stamp_classifier((stamps, metadata), training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.stamp_classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.stamp_classifier.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

        return {"loss": self.train_loss.result(), "accuracy": self.train_accuracy.result()}

    def test_step(self, data):
        (stamps, metadata), labels = data
        predictions = self.stamp_classifier((stamps, metadata), training=False)

        # Compute your custom validation loss here
        val_loss = balanced_xentropy(labels, predictions)
        
        self.val_loss.update_state(val_loss)
        self.val_accuracy.update_state(labels, predictions)

        # Convierte predicciones a clases para precisión y recall
        predicted_classes = tf.argmax(predictions, axis=-1)

        self.val_precision.update_state(labels, predicted_classes)
        self.val_recall.update_state(labels, predicted_classes)

        # F1 calculado a partir de precisión y recall
        precision = self.val_precision.result()
        recall = self.val_recall.result()
        f1 = tf.where(
            precision + recall > 0,
            2 * (precision * recall) / (precision + recall),
            0.0
        )

        return {
            "val_loss": self.val_loss.result(),
            "val_accuracy": self.val_accuracy.result(),
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        }