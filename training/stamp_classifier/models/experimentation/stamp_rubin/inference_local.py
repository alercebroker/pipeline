import tensorflow as tf
import logging
import hydra
import yaml
import os
import sys
import pandas as pd
import json  # Importamos json para guardar el mapeo de clases

from omegaconf import DictConfig, OmegaConf

# Asegúrate de que los módulos de tu proyecto sean importables
from src.data.data_loader import get_tf_datasets
from src.models.CNN_model import DynamicStampModel

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Funciones de Ayuda ---
def load_yaml(path):
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def generate_predictions(model, dataset):
    """
    Itera sobre el dataset y recopila las etiquetas verdaderas, las clases predichas
    y las probabilidades completas.
    """
    logging.info("Starting prediction generation...")
    y_true_all = []
    y_pred_classes_all = []
    y_pred_probs_all = []

    for (stamps, metadata), labels in dataset:
        # 1. Obtener las predicciones raw (logits)
        preds_raw = model((stamps, metadata), training=False)
        # 2. Convertir a probabilidades con softmax
        preds_probs = tf.nn.softmax(preds_raw, axis=1)
        # 3. Obtener la clase con la probabilidad más alta
        preds_classes = tf.argmax(preds_probs, axis=1)

        y_true_all.extend(labels.numpy())
        y_pred_classes_all.extend(preds_classes.numpy())
        y_pred_probs_all.extend(preds_probs.numpy())
    
    logging.info("Prediction generation complete.")
    return y_true_all, y_pred_classes_all, y_pred_probs_all


@hydra.main(config_path="./configs", config_name="inference_local_config", version_base=None)
def run_inference_local(config: DictConfig) -> None:
    """
    Script principal para ejecutar la inferencia y guardar los resultados en un archivo Parquet.
    """
    args = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    model_dir_path = args['model_dir_path']
    logging.info(f"Loading model and configuration from local directory: {model_dir_path}")

    # --- 1. Cargar la configuración del entrenamiento (`hparams.yaml`) ---
    hparams_path = os.path.join(model_dir_path, "hparams.yaml")
    logging.info(f"Loading hyperparameters from: {hparams_path}")
    train_args = load_yaml(hparams_path)

    # --- 2. Cargar los datos ---
    train_args['model_dir_path'] = model_dir_path 
    train_args['loader']['fold'] = args.get('fold_to_evaluate', train_args['loader']['fold'])
    logging.info(f"Loading data for fold: {train_args['loader']['fold']}")
    
    # Necesitamos capturar los OIDs del conjunto de test
    train_args['dir_data'] = '/home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/data_acquisition/rubin/data/processed/ts_stamps_v0.0.2_comm_4candmax'
    train_args['path_partition'] = '/home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/data_acquisition/rubin/data/processed/partitions/partitions_trainSN_valSN_real_eval_firstStamp_v1/partitions.parquet'
    train_ds, train_ds_for_eval, val_ds, test_ds, _, _, oids_test, dict_info = get_tf_datasets(
        batch_size=train_args['loader']['batch_size'], args=train_args
    )
    logging.info("Test dataset loaded successfully.")

    # --- 3. Cargar el modelo completo (`model.keras`) ---
    model_path = os.path.join(model_dir_path, "model.keras")
    logging.info(f"Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"DynamicStampModel": DynamicStampModel},
        compile=False
    )
    model.summary()

    # --- 4. Generar predicciones ---
    y_true, y_pred_class, y_pred_probs = generate_predictions(model, test_ds)
    
    # --- 5. Construir y guardar el DataFrame ---
    # Crear un diccionario para el DataFrame inicial
    results_data = {
        'oid': oids_test,
        'y_true': y_true,
        'y_pred': y_pred_class
    }
    results_df = pd.DataFrame(results_data)

    # Crear columnas de probabilidad para cada clase
    class_map = dict_info["dict_mapping_classes"]
    prob_cols = [f"prob_{class_map[i]}" for i in sorted(class_map.keys())]
    probs_df = pd.DataFrame(y_pred_probs, columns=prob_cols)

    # Unir todo en un solo DataFrame
    final_df = pd.concat([results_df, probs_df], axis=1)

    # Definir la ruta de salida
    output_path = args.get('output_dir', f"inference_results/{os.path.basename(model_dir_path)}")
    os.makedirs(output_path, exist_ok=True)

    # Guardar el DataFrame como Parquet
    parquet_filename = args.get('output_filename', 'predictions.parquet')
    parquet_filepath = os.path.join(output_path, parquet_filename)
    final_df.to_parquet(parquet_filepath, index=False)
    logging.info(f"Predictions saved to: {parquet_filepath}")

    # Guardar el mapeo de clases como JSON para referencia en el notebook
    class_map_path = os.path.join(output_path, "class_map.json")
    with open(class_map_path, 'w') as f:
        json.dump(class_map, f, indent=4)
    logging.info(f"Class mapping saved to: {class_map_path}")
    
    logging.info("Inference script finished successfully.")


if __name__ == "__main__":
    run_inference_local()