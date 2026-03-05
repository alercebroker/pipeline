import os
import tensorflow as tf
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from src.training.early_stopper import NoImprovementStopper
from src.utils.plots import save_confusion_matrix_and_report
from inference_mr import eval_step

class Trainer:
    def __init__(
            self, 
            model, 
            loss_object, 
            optimizer, 
            args, 
            train_ds, 
            train_ds_for_eval, 
            val_ds, 
            test_ds,
            oids_train,
            oids_val,
            oids_test,
            candid_train,
            candid_val,
            candid_test,
            test_ds_sim,
            oid_test_sim,
            test_files_ids,
            artifact_path, 
            dict_info
            ):
        
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.args = args
        self.train_ds = train_ds
        self.train_ds_for_eval = train_ds_for_eval
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.oids_train = oids_train
        self.oids_val = oids_val
        self.oids_test = oids_test
        self.candid_train = candid_train
        self.candid_val = candid_val
        self.candid_test = candid_test
        self.test_ds_sim = test_ds_sim
        self.oid_test_sim = oid_test_sim
        self.test_files_ids = test_files_ids
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
        self.train_loss = tf.keras.metrics.Mean(name='train_loss_running')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_running')
        self.eval_train_at_the_epoch_end = args['training']['eval_train_at_the_epoch_end']

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
    def train_step(self, inputs_dict, labels):
        """
        Paso de entrenamiento adaptado para recibir un diccionario o tupla.
        """
        with tf.GradientTape() as tape:
            # El modelo ya sabe manejar si inputs_dict es un dict o una tupla
            predictions = self.model(inputs_dict, training=True)
            loss = self.loss_object(labels, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

    def val_test_step(self, dataset, iteration, file_writer):
        """
        Evaluación durante el entrenamiento.
        """
        prediction_list, label_list = [], []
        loss_list = []
        
        if dataset is None: 
            return 0.0, 0.0, 0.0

        # Iteramos sobre el dataset. 
        # En modo MultiRes: batch es (inputs_dict, labels)
        # En modo Legacy: batch es ((images, meta), labels)
        # Python desempaqueta 'inputs' correctamente en ambos casos (como objeto único o tupla)
        for inputs, labels in dataset:
            predictions = self.model(inputs, training=False)
            loss = self.loss_object(labels, predictions)
            loss_list.append(loss)
            prediction_list.append(predictions)
            label_list.append(labels)

        xentropy = tf.reduce_mean(tf.stack(loss_list))
        labels = tf.concat(label_list, axis=0)
        predictions = tf.concat(prediction_list, axis=0)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.numpy(), predictions.numpy().argmax(axis=1), average='macro'
        )
        
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

            # --- CORRECCIÓN AQUÍ ---
            # El dataset devuelve (inputs, labels).
            # Si es legacy, inputs es una tupla (img, meta).
            # Si es multires, inputs es un diccionario {'main': ..., 'level0': ...}.
            # No intentamos desempaquetar inputs aquí, se lo pasamos directo a train_step.
            inputs_batch, y_batch = training_batch
            
            self.train_step(inputs_batch, y_batch)

            # Log de entrenamiento por iteración
            if iteration % log_frequency == 0 and iteration != 0:
                train_loss_running = self.train_loss.result().numpy()
                train_acc_running = self.train_accuracy.result().numpy()

                with self.train_writer.as_default():
                    tf.summary.scalar('loss_running', train_loss_running, step=iteration)
                    tf.summary.scalar('accuracy_running', train_acc_running, step=iteration)

                mlflow.log_metric("train_loss_running", train_loss_running, step=iteration)
                mlflow.log_metric("train_accuracy_running", train_acc_running, step=iteration)

            # Validación
            if iteration % val_frequency == 0:
                train_loss = self.train_loss.result().numpy()
                train_acc = self.train_accuracy.result().numpy()
                train_f1 = 0.0
                
                # --- Evaluación del Set de Entrenamiento ---
                if self.eval_train_at_the_epoch_end:
                    train_f1, train_loss, train_acc = self.val_test_step(self.train_ds_for_eval, iteration, self.train_writer)
                    mlflow.log_metric("train_loss", train_loss, step=iteration)
                    mlflow.log_metric("train_accuracy", train_acc, step=iteration)
                    mlflow.log_metric("train_f1", train_f1, step=iteration)
                else:
                    mlflow.log_metric("train_loss_epoch_avg", train_loss, step=iteration)
                    mlflow.log_metric("train_accuracy_epoch_avg", train_acc, step=iteration)

                # --- Evaluación del Set de Validación ---
                val_f1, val_loss, val_acc = self.val_test_step(self.val_ds, iteration, self.val_writer)
                mlflow.log_metric("val_loss", val_loss, step=iteration)
                mlflow.log_metric("val_accuracy", val_acc, step=iteration)
                mlflow.log_metric("val_f1", val_f1, step=iteration)

                train_log_str = (
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
                    if self.eval_train_at_the_epoch_end
                    else f"Train Loss (avg): {train_loss:.4f} | Train Acc (avg): {train_acc:.4f}"
                )
                
                print(
                    f"[{iteration:05d}] {train_log_str} | "
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

    def _predict_and_save_dataframe(self, dataset, oids, candid, dataset_name):
        print(f"\n[Evaluator] Generating predictions for '{dataset_name}' set...")
        if dataset is None or oids is None or len(oids) == 0:
            print(f"[Warning] Dataset or OIDs for '{dataset_name}' not provided. Skipping.")
            return None, None, None
        
        # NOTA: eval_step (importado de inference) debe soportar también el formato de diccionario.
        # Si eval_step falla, probablemente necesite una adaptación similar a train_step.
        # Asumimos que eval_step usa model.predict() o similar que acepta el input del dataset.
        _, _, _, _, labels_int, predictions_int, probs = eval_step(self.model, dataset)
        
        df = pd.DataFrame(
            {
                'oid': oids, 
                'measurement_id': candid, 
                'true_label_int': labels_int, 
                'predicted_label_int': predictions_int
            }
        )
        class_map = self.dict_info['dict_mapping_classes']
        df['true_label'] = df['true_label_int'].map(class_map)
        df['predicted_label'] = df['predicted_label_int'].map(class_map)
        predicted_probs = probs[np.arange(len(probs)), predictions_int]
        df['predicted_probability'] = predicted_probs
        for class_idx, class_name in class_map.items():
            df[f'prob_{class_name}'] = probs[:, class_idx]
        predictions_path = os.path.join(self.artifact_path, f"{dataset_name}_predictions.csv")
        df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)
        print(f"Saved predictions to {predictions_path}")
        return labels_int, predictions_int, probs

    def finalize_and_save_results(self):
        self._predict_and_save_dataframe(self.train_ds_for_eval, self.oids_train, self.candid_train, "train")
        self._predict_and_save_dataframe(self.val_ds, self.oids_val, self.candid_val, "val")
        self._predict_and_save_dataframe(self.test_ds_sim, self.oid_test_sim, self.test_files_ids, "test_sim")
        
        test_labels_int, test_predictions_int, test_probs = self._predict_and_save_dataframe(self.test_ds, self.oids_test, self.candid_test, "test")

        if test_labels_int is None:
            print("[Error] Could not evaluate test set. Final metrics will not be available.")
            return

        class_map = self.dict_info['dict_mapping_classes']
        class_names = list(class_map.values())
        test_labels_str = [class_map[x] for x in test_labels_int]
        test_predictions_str = [class_map[x] for x in test_predictions_int]

        import json
        print("[Evaluator] Saving sample predictions JSON...")
        # Guardar un ejemplo simple
        results_dict = {
            f"ejemplo_{i}": {
                "class": class_names[pred],
                "probability": float(test_probs[i][pred])
            }
            for i, pred in enumerate(test_predictions_int[:10])
        }

        json_path = os.path.join(self.artifact_path, 'deployment_checks', "predictions.json")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        mlflow.log_artifact(json_path)
        
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels_str, test_predictions_str, average='macro')
        
        print(f"\n✅ Final Evaluation (Test Set)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-score:  {f1:.4f}")
        print(f"\n🏅 Best Validation {self.monitor.upper()}: {self.best_metric:.4f}")

        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        
        path_save_metrics = os.path.join(self.artifact_path, "metrics")
        os.makedirs(path_save_metrics, exist_ok=True)
        save_confusion_matrix_and_report(test_labels_str, test_predictions_str, path_save_metrics, class_names=class_names)
        
        model_path = os.path.join(self.artifact_path, "model.keras")
        self.model.save(model_path)
        print(f"\n🚀 Model saved to {model_path}")

#    def evaluate_and_save(self):
#        # La evaluación ya se hace aquí, solo necesitamos usar los resultados
#        _, _, _, _, test_labels_int, test_predictions_int, probs = eval_step(self.model, self.test_ds)
#
#        #print('test_labels_int:', test_labels_int)
#        #print('test_predictions_int:', test_predictions_int)
#        #exit()
#        
#        class_names = list(self.dict_info['dict_mapping_classes'].values())
#        
#        # --- INICIO DE LA NUEVA LÓGICA PARA CREAR EL DATAFRAME ---
#        # 1. Crear el DataFrame inicial con OIDs y etiquetas/predicciones enteras
#        df_predictions = pd.DataFrame({
#            'oid': self.oids_test,
#            'true_label_int': test_labels_int,
#            'predicted_label_int': test_predictions_int
#        })
#
#        # 2. Mapear los enteros a los nombres de las clases
#        df_predictions['true_label'] = df_predictions['true_label_int'].map(self.dict_info['dict_mapping_classes'])
#        df_predictions['predicted_label'] = df_predictions['predicted_label_int'].map(self.dict_info['dict_mapping_classes'])
#        
#        # 3. Extraer la probabilidad de la clase predicha
#        # (Esto usa indexación avanzada de NumPy para obtener la probabilidad correcta para cada fila)
#        predicted_probs = probs[np.arange(len(probs)), test_predictions_int]
#        df_predictions['predicted_probability'] = predicted_probs
#
#        # 4. (Opcional pero muy recomendado) Añadir las probabilidades de TODAS las clases
#        for class_idx, class_name in self.dict_info['dict_mapping_classes'].items():
#            df_predictions[f'prob_{class_name}'] = probs[:, class_idx]
#
#        # 5. Guardar el DataFrame como un artefacto CSV
#        predictions_path = os.path.join(self.artifact_path, "test_predictions.csv")
#        df_predictions.to_csv(predictions_path, index=False)
#        mlflow.log_artifact(predictions_path)
#
#        # --- FIN DE LA NUEVA LÓGICA ---
#
#        # El resto de tu código de evaluación y guardado puede continuar
#        test_labels_str = [self.dict_info['dict_mapping_classes'][x] for x in test_labels_int]
#        test_predictions_str = [self.dict_info['dict_mapping_classes'][x] for x in test_predictions_int]
#
#
#        # ... (tu código para el JSON de ejemplo, que ahora es redundante pero puedes mantenerlo)
#        import json
#        results_dict = {
#            f"ejemplo_{i}": {
#                "class": class_names[pred],
#                "probability": float(probs[i][pred])
#            }
#            for i, pred in enumerate(test_predictions_int[:10])
#        }
#        path_save_metrics = os.path.join(self.artifact_path, "metrics")
#        os.makedirs(path_save_metrics, exist_ok=True)
#        with open(os.path.join(path_save_metrics, "predictions.json"), "w") as f:
#            json.dump(results_dict, f, indent=4)
#        ############
#
#        
#        # Guardar métricas finales en MLflow
#        precision, recall, f1, _ = precision_recall_fscore_support(
#            test_labels_str, test_predictions_str, average='macro'
#        )
#        
#        print(f"\n✅ Final Evaluation (Test Set)")
#        print(f"Test Precision: {precision:.4f}")
#        print(f"Test Recall:    {recall:.4f}")
#        print(f"Test F1-score:  {f1:.4f}")
#        print(f"\n🏅 Best Validation {self.monitor.upper()}: {self.best_metric:.4f}")
#
#        mlflow.log_metric("test_f1", f1)
#        mlflow.log_metric("test_precision", precision)
#        mlflow.log_metric("test_recall", recall)
#        
#        path_save_metrics = os.path.join(self.artifact_path, "metrics")
#        os.makedirs(path_save_metrics, exist_ok=True)
#        save_confusion_matrix_and_report(test_labels_str, test_predictions_str, path_save_metrics, class_names=class_names)
#
#        self.model.save(os.path.join(self.artifact_path, "model.keras"))
#