import os
import tensorflow as tf
import mlflow
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from src.training.losses import balanced_xentropy
from src.training.early_stopper import NoImprovementStopper
from src.utils.plots import save_confusion_matrix_and_report
from inference import eval_step

class Trainer:
    def __init__(self, model, loss_object, optimizer, args, train_ds, val_ds, test_ds, artifact_path, dict_info):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.args = args
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.artifact_path = artifact_path
        self.dict_info = dict_info
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        self.model.optimizer = self.optimizer
        self.lr_scheduler.set_model(self.model)


        # Métricas de entrenamiento
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # Logging de TensorBoard
        logdir = os.path.join(artifact_path, 'logs')
        self.train_writer = tf.summary.create_file_writer(logdir + '/train')
        self.val_writer = tf.summary.create_file_writer(logdir + '/val')
        self.test_writer = tf.summary.create_file_writer(logdir + '/test')

        # Early stopping
        self.monitor = args['training']['monitor']
        self.stopper_mode = 'min' if self.monitor == 'loss' else 'max'
        self.stopper = NoImprovementStopper(num_steps=10, mode=self.stopper_mode)
        self.best_metric = float('inf') if self.stopper_mode == 'min' else -float('inf')
        self.best_weights = None

    @tf.function
    def train_step(self, stamps, metadata, labels):
        with tf.GradientTape() as tape:
            predictions = self.model((stamps, metadata), training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

    def val_test_step(self, dataset, iteration, file_writer):
        prediction_list, label_list = [], []
        loss_list = []
        for (samples, md), labels in dataset:
            predictions = self.model((samples, md), training=False)
            loss = self.loss_object(labels, predictions)  # Usar la misma función de pérdida que en entrenamiento
            loss_list.append(loss)
            prediction_list.append(predictions)
            label_list.append(labels)

        xentropy = tf.reduce_mean(tf.stack(loss_list))
        labels = tf.concat(label_list, axis=0)
        predictions = tf.concat(prediction_list, axis=0)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.numpy(), predictions.numpy().argmax(axis=1), average='macro'
        )
        #xentropy = balanced_xentropy(labels, predictions)

        # ✅ Calcular accuracy manualmente
        val_accuracy = np.mean((predictions.numpy().argmax(axis=1) == labels.numpy()).astype(np.float32))

        with file_writer.as_default():
            tf.summary.scalar('precision', precision, step=iteration)
            tf.summary.scalar('recall', recall, step=iteration)
            tf.summary.scalar('f1', f1, step=iteration)
            tf.summary.scalar('loss', xentropy, step=iteration)
            tf.summary.scalar('accuracy', val_accuracy, step=iteration)

        return f1, xentropy, val_accuracy

    def fit(self, log_frequency=50, val_frequency=500, max_iterations=1_000_000):
        
        for iteration, training_batch in enumerate(self.train_ds):
            if iteration >= max_iterations:
                print(f"[Trainer] Reached max_iterations = {max_iterations}")
                break

            (x_batch, md_batch), y_batch = training_batch
            self.train_step(x_batch, md_batch, y_batch)

            # Log de entrenamiento por iteración
            if iteration % log_frequency == 0 and iteration != 0:
                train_loss_val = self.train_loss.result().numpy()
                train_acc_val = self.train_accuracy.result().numpy()

                with self.train_writer.as_default():
                    tf.summary.scalar('loss', train_loss_val, step=iteration)
                    tf.summary.scalar('accuracy', train_acc_val, step=iteration)

                mlflow.log_metric("train_loss", train_loss_val, step=iteration)
                mlflow.log_metric("train_accuracy", train_acc_val, step=iteration)

            # Validación
            if iteration % val_frequency == 0:
                # Calcular métricas de validación
                val_f1, val_loss, val_acc = self.val_test_step(self.val_ds, iteration, self.val_writer)

                # Calcular métricas de entrenamiento actuales
                train_loss_val = self.train_loss.result().numpy()
                train_acc_val = self.train_accuracy.result().numpy()

                # Loguear todo en MLflow
                mlflow.log_metric("train_loss", train_loss_val, step=iteration)
                mlflow.log_metric("train_accuracy", train_acc_val, step=iteration)
                mlflow.log_metric("val_loss", val_loss, step=iteration)
                mlflow.log_metric("val_accuracy", val_acc, step=iteration)
                mlflow.log_metric("val_f1", val_f1, step=iteration)

                # Mostrar en consola
                print(
                    f"[{iteration:05d}] "
                    f"Train Loss: {train_loss_val:.4f} | Train Acc: {train_acc_val:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )

                # Guardar el mejor modelo
                val_metric = val_loss if self.monitor == 'loss' else val_f1
                self.lr_scheduler.on_epoch_end(iteration, logs={self.lr_scheduler.monitor: val_loss})

                if ((self.stopper_mode == 'min' and val_metric < self.best_metric) or 
                    (self.stopper_mode == 'max' and val_metric > self.best_metric)):
                    self.best_metric = val_metric
                    self.best_weights = self.model.get_weights()

                if self.stopper.should_break(val_metric):
                    print(f"[Trainer] Early stopping triggered at iteration {iteration}")
                    break

            self.train_loss.reset_state()
            self.train_accuracy.reset_state()

        # Cargar pesos óptimos
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

        self.train_writer.flush()
        self.val_writer.flush()
        self.test_writer.flush()

    def evaluate_and_save(self):
        _, _, _, _, test_labels, test_predictions = eval_step(self.model, self.test_ds)
        test_labels = [self.dict_info['dict_mapping_classes'][x] for x in test_labels]
        test_predictions = [self.dict_info['dict_mapping_classes'][x] for x in test_predictions]
        class_names = list(self.dict_info['dict_mapping_classes'].values())

        print(self.dict_info['dict_mapping_classes'])
        
        # Guardar métricas finales en MLflow
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average='macro'
        )
        
        # Mostrar en consola (para los sprints)
        print(f"\n✅ Final Evaluation (Test Set)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-score:  {f1:.4f}")

        print(f"\n🏅 Best Validation {self.monitor.upper()}: {self.best_metric:.4f}")

        # Guardar en MLflow
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)

        path_save_metrics = os.path.join(self.artifact_path, "metrics")
        os.makedirs(path_save_metrics, exist_ok=True)
        save_confusion_matrix_and_report(test_labels, test_predictions, path_save_metrics, class_names=class_names)

        self.model.save(os.path.join(self.artifact_path, "model.keras"))
