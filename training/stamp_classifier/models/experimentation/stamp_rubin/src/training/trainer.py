import os
import tensorflow as tf
import mlflow
import numpy as np
import pandas as pd
import json
import joblib
import yaml

from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from src.training.losses import balanced_xentropy
from src.training.early_stopper import NoImprovementStopper
from src.utils.plots import save_confusion_matrix_and_report, log_metrics_to_mlflow
from inference import eval_step


# --- Clase para serializar JSON ---
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyAwareJSONEncoder, self).default(obj)


class ModelTemperature:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, y_true):
        # Función objetivo: Minimizar NLL (Negative Log Likelihood)
        def nll_loss(t):
            if t <= 0: return 1e9
            scaled_logits = logits / t
            probs = softmax(scaled_logits, axis=1)
            return log_loss(y_true, probs)

        # Optimización escalar
        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 5.0)], method='L-BFGS-B')
        self.T = float(res.x[0])
        return self

    def predict_proba(self, logits):
        """
        Aplica la temperatura a los logits y devuelve probabilidades (Softmax).
        """
        return softmax(logits / self.T, axis=1)
    

class Trainer:
    def __init__(
            self, model, loss_object, optimizer, args, 
            train_ds, train_ds_for_eval, val_ds, test_ds,
            oids_train, oids_val, oids_test,
            candid_train, candid_val, candid_test,
            ra_train, dec_train, ra_val, dec_val, 
            ra_test, dec_test, ra_test_sim, dec_test_sim,
            test_ds_sim, oid_test_sim, test_files_ids,
            artifact_path, dict_info
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

        self.ra_train = ra_train
        self.dec_train = dec_train
        self.ra_val = ra_val
        self.dec_val = dec_val
        self.ra_test = ra_test
        self.dec_test = dec_test
        self.ra_test_sim = ra_test_sim
        self.dec_test_sim = dec_test_sim

        self.test_ds_sim = test_ds_sim
        self.oid_test_sim = oid_test_sim
        self.test_files_ids = test_files_ids
        self.artifact_path = artifact_path
        self.dict_info = dict_info

        # Scheduler para bajar el Learning Rate (usando val_loss por estabilidad de gradiente)
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
        logdir = os.path.join(artifact_path, 'training/logs')
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
        f05 = fbeta_score(labels.numpy(), predictions.numpy().argmax(axis=1), beta=0.5, average='macro', zero_division=0)
        #xentropy = balanced_xentropy(labels, predictions)

        # ✅ Calcular accuracy manualmente
        val_accuracy = np.mean((predictions.numpy().argmax(axis=1) == labels.numpy()).astype(np.float32))

        with file_writer.as_default():
            tf.summary.scalar('precision', precision, step=iteration)
            tf.summary.scalar('recall', recall, step=iteration)
            tf.summary.scalar('f1', f1, step=iteration)
            tf.summary.scalar('f05', f05, step=iteration)
            tf.summary.scalar('loss', xentropy, step=iteration)
            tf.summary.scalar('accuracy', val_accuracy, step=iteration)

        return f1, f05, xentropy, val_accuracy

    def fit(self, log_frequency=50, val_frequency=500, max_iterations=1_000_000):
        print(self.train_ds)
        for iteration, training_batch in enumerate(self.train_ds):
            if iteration >= max_iterations:
                print(f"[Trainer] Reached max_iterations = {max_iterations}")
                break

            (x_batch, md_batch), y_batch = training_batch
            self.train_step(x_batch, md_batch, y_batch)

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
                # Evaluate on the full training set
                train_loss = self.train_loss.result().numpy()
                train_acc = self.train_accuracy.result().numpy()
                
                # --- Evaluación Opcional del Set de Entrenamiento ---
                if self.eval_train_at_the_epoch_end:
                    # Desempaquetamos f05
                    train_f1, train_f05, train_loss, train_acc = self.val_test_step(self.train_ds_for_eval, iteration, self.train_writer)
                    mlflow.log_metric("train_loss", train_loss, step=iteration)
                    mlflow.log_metric("train_accuracy", train_acc, step=iteration)
                    mlflow.log_metric("train_f1", train_f1, step=iteration)
                    mlflow.log_metric("train_f05", train_f05, step=iteration)
                else:
                    train_f1, train_f05 = 0.0, 0.0
                    mlflow.log_metric("train_loss_epoch_avg", train_loss, step=iteration)
                    mlflow.log_metric("train_accuracy_epoch_avg", train_acc, step=iteration)

                # --- Evaluación del Set de Validación (siempre se ejecuta) ---
                # Desempaquetamos f05
                val_f1, val_f05, val_loss, val_acc = self.val_test_step(self.val_ds, iteration, self.val_writer)
                
                mlflow.log_metric("val_loss", val_loss, step=iteration)
                mlflow.log_metric("val_accuracy", val_acc, step=iteration)
                mlflow.log_metric("val_f1", val_f1, step=iteration)
                mlflow.log_metric("val_f05", val_f05, step=iteration)

                train_log_str = (
                    f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}"
                    if self.eval_train_at_the_epoch_end
                    else f"Train Loss (avg): {train_loss:.4f}"
                )
                
                print(
                    f"[{iteration:05d}] {train_log_str} | "
                    f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val F0.5: {val_f05:.4f}"
                )

                # Scheduler: Reducir LR basado en Loss (estabilidad)
                self.lr_scheduler.on_epoch_end(iteration, logs={'val_loss': val_loss})

                # --- SELECCIÓN DE MÉTRICA PARA EARLY STOPPING ---
                if self.monitor == 'loss':
                    val_metric = val_loss
                elif self.monitor == 'f05':
                    val_metric = val_f05
                elif self.monitor == 'f1':
                    val_metric = val_f1
                elif self.monitor == 'accuracy':
                    val_metric = val_acc
                else:
                    val_metric = val_loss # Default fallback

                # Guardar el mejor modelo
                if ((self.stopper_mode == 'min' and val_metric < self.best_metric) or 
                    (self.stopper_mode == 'max' and val_metric > self.best_metric)):
                    self.best_metric = val_metric
                    self.best_weights = self.model.get_weights()
                    print(f"🌟 Best model updated based on {self.monitor}: {self.best_metric:.4f}")

                if self.stopper.should_break(val_metric):
                    print(f"[Trainer] Early stopping triggered at iteration {iteration} (Monitor: {self.monitor})")
                    break

            self.train_loss.reset_state()
            self.train_accuracy.reset_state()

        # Cargar pesos óptimos
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

        self.train_writer.flush()
        self.val_writer.flush()
        self.test_writer.flush()

    def calibrate_and_save(self):
        """
        Ejecuta Temperature Scaling sobre el set de validación.
        Actualiza hparams.yaml con la temperatura óptima.
        NO guarda el objeto pickle.
        """
        print("\n[Calibration] Starting Temperature Scaling process on Validation Set...")
        
        # 1. Obtener Logits y Labels de Validación
        y_logits_list = []
        y_true_list = []
        
        for (samples, md), labels in self.val_ds:
            logits = self.model((samples, md), training=False)
            y_logits_list.append(logits)
            y_true_list.append(labels)
            
        y_logits_val = tf.concat(y_logits_list, axis=0).numpy()
        y_true_val = tf.concat(y_true_list, axis=0).numpy()
        
        # 2. Entrenar Temperature Scaler
        temp_scaler = ModelTemperature()
        temp_scaler.fit(y_logits_val, y_true_val)
        
        print(f"[Calibration] Optimal Temperature found: T = {temp_scaler.T:.4f}")
        mlflow.log_metric("temperature_scaling_T", temp_scaler.T)
        
        # 3. Actualizar hparams.yaml (Registro en texto plano)
        hparams_path = os.path.join(self.artifact_path, "hparams.yaml")
        if os.path.exists(hparams_path):
            with open(hparams_path, "r") as f:
                hparams = yaml.safe_load(f)
            
            # Guardamos la config de calibración
            hparams['calibration'] = {
                'method': 'temperature_scaling',
                'temperature': float(temp_scaler.T)
            }
            
            with open(hparams_path, "w") as f:
                yaml.dump(hparams, f, sort_keys=False)
            
            # Re-loguear el artifact actualizado
            mlflow.log_artifact(hparams_path)
            print(f"[Calibration] hparams.yaml updated with T={temp_scaler.T:.4f}")
            
        return temp_scaler
    
    def _predict_and_save_dataframe(self, dataset, oids, candid, ra, dec, dataset_name, temp_scaler=None):
        print(f"\n[Evaluator] Generating predictions for '{dataset_name}' set...")
        if dataset is None or oids is None:
            print(f"[Warning] Dataset or OIDs for '{dataset_name}' not provided. Skipping.")
            return None, None, None, None
        
        # 1. Obtener predicciones RAW del modelo (Probabilidades con Softmax)
        # eval_step ya devuelve probs (softmax applied)
        _, _, _, _, labels_int, predictions_int_raw, logits_raw = eval_step(self.model, dataset)
        
        # Calcular probabilidades RAW reales usando Softmax
        probs_raw = softmax(logits_raw, axis=1)

        # 2. Calcular predicciones CALIBRADAS (Temperature Scaling)
        if temp_scaler:
            probs_cal = temp_scaler.predict_proba(logits_raw) # <-- CAMBIO
            predictions_int_cal = np.argmax(probs_cal, axis=1)
        else:
            # Si no hay scaler, las calibradas son iguales a las raw
            probs_cal = probs_raw
            predictions_int_cal = predictions_int_raw

        # 3. Construir DataFrame
        df = pd.DataFrame({
            'oid': oids, 
            'measurement_id': candid,
            'ra': ra,   
            'dec': dec,  
            'true_label_int': labels_int, 
            'predicted_label_int_raw': predictions_int_raw,
            'predicted_label_int_cal': predictions_int_cal
        })
        
        class_map = self.dict_info['dict_mapping_classes']
        
        # Mapear etiquetas a texto
        df['true_label'] = df['true_label_int'].map(class_map)
        df['predicted_label_raw'] = df['predicted_label_int_raw'].map(class_map)
        df['predicted_label_cal'] = df['predicted_label_int_cal'].map(class_map)
        
        # Probabilidades RAW
        for class_idx, class_name in class_map.items():
            df[f'prob_{class_name}_raw'] = probs_raw[:, class_idx]
            
        # Probabilidades CALIBRADAS
        for class_idx, class_name in class_map.items():
            df[f'prob_{class_name}_cal'] = probs_cal[:, class_idx]

        # Guardar CSV
        csv_name = f"{dataset_name}_predictions.csv"
        df.to_csv(csv_name, index=False)
        mlflow.log_artifact(csv_name, artifact_path="training/predictions")
        os.remove(csv_name)

        #predictions_path = os.path.join(self.artifact_path, "training", "predictions", f"{dataset_name}_predictions.csv")
        #df.to_csv(predictions_path, index=False)
        #mlflow.log_artifact(f"{dataset_name}_predictions.csv", artifact_path="training/predictions")
        #print(f"Saved predictions to {predictions_path}")
        
        return labels_int, predictions_int_cal, probs_cal, df
    
    def generate_reference_summary(self, df_train, df_val, df_test):
        print("\n[Monitoring] Generating Reference Summary (statistics)...")
        
        # 1. Unificar DataFrames y limpiar nombres de clases
        # Asumimos que los DataFrames ya tienen las columnas 'prob_AGN_cal', 'predicted_label_cal', etc.
        df_train['subset'] = 'train'
        df_val['subset'] = 'val'
        df_test['subset'] = 'test'
        
        reference_df_wide = pd.concat([df_train, df_val, df_test], ignore_index=True)
        
        # 2. Configuración
        HISTOGRAM_BINS_1D = 25
        SKY_GRID_BINS_RA = 64   
        SKY_GRID_BINS_DEC = 32  

        reference_summary = {
            "class_summary": {}, 
            "class_statistics": {}, 
            "hist_data": {}, 
            "heatmap_data": {}
        }
        
        # 3. Resumen de Clases Reales (Prior Distribution)
        class_counts = reference_df_wide['true_label'].value_counts().to_dict()
        total_count = sum(class_counts.values())
        class_fractions = {k: v / total_count for k, v in class_counts.items()}
        reference_summary['class_summary'] = {'counts': class_counts, 'fractions': class_fractions}
        
        # 4. Estadísticas por Clase PREDICHA
        # Identificar columnas de probabilidad
        prob_cols = [c for c in reference_df_wide.columns if c.startswith('prob_') and c.endswith('_cal')]
        print(f"[Monitoring] Probability columns found: {prob_cols}")
        
        # Calcular confianza y entropía
        reference_df_wide['confidence'] = reference_df_wide[prob_cols].max(axis=1)
        reference_df_wide['entropy'] = reference_df_wide[prob_cols].apply(lambda row: entropy(row, base=2), axis=1)
        
        grouped_pred = reference_df_wide.groupby('predicted_label_cal')
        stats_dict = {}
        
        for class_name, group_df in grouped_pred:
            stats_dict[class_name] = {
                'alert_count': len(group_df),
                'confidence_mean': group_df['confidence'].mean(),
                'confidence_std': group_df['confidence'].std(ddof=0),
                'confidence_min': group_df['confidence'].min(),
                'confidence_max': group_df['confidence'].max(),
                'entropy_mean': group_df['entropy'].mean()
            }
        reference_summary['class_statistics'] = stats_dict
        
        # 5. Histogramas 1D
        for class_name, group_df in grouped_pred:
            # Histograma Confianza
            conf_counts, conf_bins = np.histogram(group_df['confidence'], bins=HISTOGRAM_BINS_1D, range=(0, 1))
            reference_summary['hist_data'][f"confidence|{class_name}"] = {
                'counts': conf_counts.tolist(), 'bins': conf_bins.tolist()
            }
            # Histograma Entropía
            max_ent = np.log2(len(prob_cols))
            ent_counts, ent_bins = np.histogram(group_df['entropy'], bins=HISTOGRAM_BINS_1D, range=(0, max_ent))
            reference_summary['hist_data'][f"entropy|{class_name}"] = {
                'counts': ent_counts.tolist(), 'bins': ent_bins.tolist()
            }
            
        # Distribución Global por Clase (Probabilidades raw de cada neurona de salida)
        for prob_col in prob_cols:
            clean_name = prob_col.replace('prob_', '').replace('_cal', '')
            prob_counts, prob_bins = np.histogram(reference_df_wide[prob_col], bins=HISTOGRAM_BINS_1D, range=(0, 1))
            reference_summary['hist_data'][f"prob_{clean_name}|ALL"] = {
                'counts': prob_counts.tolist(), 'bins': prob_bins.tolist()
            }

        if 'ra' in reference_df_wide.columns and not reference_df_wide['ra'].isnull().all():
            print("[Monitoring] Calculating heatmaps for the reference set...")
            
            # Filtramos nulos de ra/dec y agrupamos por la clase predicha calibrada
            grouped_sky = reference_df_wide.dropna(subset=['ra', 'dec']).groupby('predicted_label_cal')
            
            ra_bins = np.linspace(0, 360, SKY_GRID_BINS_RA + 1)
            dec_bins = np.linspace(-90, 90, SKY_GRID_BINS_DEC + 1)
            
            for class_name, group_df in grouped_sky:
                heatmap_counts, _, _ = np.histogram2d(group_df['ra'], group_df['dec'], bins=[ra_bins, dec_bins])
                
                # Convertimos a listas nativas de python con .tolist() para compatibilidad directa con JSON
                reference_summary['heatmap_data'][class_name] = {
                    'counts': heatmap_counts.tolist(), 
                    'ra_bins': ra_bins.tolist(), 
                    'dec_bins': dec_bins.tolist()
                }

        # 6. Guardar JSON
        mlflow.log_dict(reference_summary, "monitoring/reference_summary.json")

        #summary_path = os.path.join(self.artifact_path, "monitoring", "reference_summary.json")
        #with open(summary_path, 'w') as f:
        #    json.dump(reference_summary, f, indent=4, cls=NumpyAwareJSONEncoder)
        #
        #mlflow.log_artifact(summary_path, artifact_path="monitoring")
        #print(f"[Monitoring] Saved reference_summary.json to {summary_path}")
    
    def finalize_and_save_results(self):
        temp_scaler = self.calibrate_and_save()
        
        # 2. Generar predicciones para todos los sets (pasando el temp_scaler)
        _, _, _, df_train = self._predict_and_save_dataframe(
            self.train_ds_for_eval, self.oids_train, self.candid_train, self.ra_train, self.dec_train, "train", temp_scaler=None
        )
        _, _, _, df_val = self._predict_and_save_dataframe(
            self.val_ds, self.oids_val, self.candid_val, self.ra_val, self.dec_val, "val", temp_scaler=None
        )        
        test_labels_int, test_predictions_int, test_probs, df_test = self._predict_and_save_dataframe(
            self.test_ds, self.oids_test, self.candid_test, self.ra_test, self.dec_test, "test", temp_scaler
        )
        _, _, _, _ = self._predict_and_save_dataframe(
            self.test_ds_sim, self.oid_test_sim, self.test_files_ids, self.ra_test_sim, self.dec_test_sim, "test_sim", temp_scaler=None
        )

        if test_labels_int is None:
            print("[Error] Could not evaluate test set. Final metrics will not be available.")
            return

        # 3. GENERAR RESUMEN DE REFERENCIA (NUEVO)
        if df_train is not None and df_val is not None and df_test is not None:
            self.generate_reference_summary(df_train, df_val, df_test)

        class_map = self.dict_info['dict_mapping_classes']
        class_names = list(class_map.values())
        test_labels_str = [class_map[x] for x in test_labels_int]
        test_predictions_str = [class_map[x] for x in test_predictions_int]

        import json
        print("[Evaluator] Saving sample predictions JSON...")
        
        # Tomamos 10 ejemplos del DataFrame de Test ya construido
        sample_df = df_test.head(10)
        results_dict = {}
        
        for _, row in sample_df.iterrows():
            # Clave: OID del objeto
            oid_key = str(row['oid'])
            
            # Valor: Detalles
            pred_class = row['predicted_label_cal']
            
            # Probabilidad de la clase predicha (calibrada)
            prob_col_name = f"prob_{pred_class}_cal"
            prob_val = row[prob_col_name]
            
            results_dict[oid_key] = {
                "candid": str(row['measurement_id']),
                "class": pred_class,
                "probability": float(prob_val)
            }

        mlflow.log_dict(results_dict, "deployment_checks/predictions.json")

        #json_path = os.path.join(self.artifact_path, 'deployment_checks', "predictions.json")
        #with open(json_path, "w") as f:
        #    json.dump(results_dict, f, indent=4)
        #mlflow.log_artifact("predictions.json", artifact_path="deployment_checks")
        #mlflow.log_artifact(json_path)
        
        # Métricas finales
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels_str, test_predictions_str, average='macro', zero_division=0)
        f05 = fbeta_score(test_labels_str, test_predictions_str, beta=0.5, average='macro', zero_division=0)
        
        print(f"\n✅ Final Evaluation (Test Set)")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-score:  {f1:.4f}")
        print(f"Test F0.5-score:{f05:.4f} (Used for purity)")
        print(f"\n🏅 Best Validation {self.monitor.upper()}: {self.best_metric:.4f}")

        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_f05", f05)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        
        # 2. Log de artefactos (gráficas y reportes)
        #path_save_metrics = os.path.join(self.artifact_path, "training", "metrics")
        #os.makedirs(path_save_metrics, exist_ok=True)
        log_metrics_to_mlflow(test_labels_str, test_predictions_str, class_names=class_names)
        
        # 3. Guardar el modelo
        # Esto esta por mientras, ya que los otros codigos llaman el modelo del artefacto
        model_path = os.path.join(self.artifact_path, "model.keras")
        self.model.save(model_path)

        #print("\n🚀 Logging model to MLflow...")
        #mlflow.tensorflow.log_model(self.model, artifact_path="model")

        #print("\n🚀 Logging model to MLflow...")
        #try:
        #    # Obtenemos un lote: ((images, metadata), labels)
        #    batch = next(iter(self.test_ds.take(1)))
        #    inputs_batch = batch[0]
        #    
        #    # MLflow para multi-input prefiere un DICCIONARIO
        #    # Usamos los nombres de las capas de entrada si es posible, o genéricos
        #    input_example = {
        #        "input_images": inputs_batch[0].numpy()[:1],
        #        "input_metadata": inputs_batch[1].numpy()[:1]
        #    }
        #    
        #    # Generar la firma con el diccionario
        #    sample_output = self.model.predict(inputs_batch, verbose=0)
        #    signature = infer_signature(input_example, sample_output)
        #    
        #    # Log del modelo
        #    mlflow.tensorflow.log_model(
        #        model=self.model,
        #        artifact_path="model", # Sigue usando artifact_path, el warning es interno de MLflow
        #        signature=signature,
        #        input_example=input_example
        #    )
        #    print("✅ Model logged successfully with dictionary signature.")
        #    
        #except Exception as e:
        #    print(f"⚠️ Could not log model with signature: {e}")
        #    # Si falla la firma, guardamos el modelo tal cual para no perder el entrenamiento
        #    mlflow.tensorflow.log_model(self.model, artifact_path="model")


        #print(f"\n🚀 Model saved to {model_path}")













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