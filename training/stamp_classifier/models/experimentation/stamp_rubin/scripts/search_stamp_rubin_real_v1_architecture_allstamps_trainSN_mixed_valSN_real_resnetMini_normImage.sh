#!/bin/bash
set -e # Salir inmediatamente si un comando falla.

# --- Configuración de Hydra ---
# Exporta las variables de entorno para que Hydra encuentre el archivo de config base.
export HYDRA_CONFIG_PATH="./configs"
export HYDRA_CONFIG_NAME="cnn_config_v1_real_rubin_allstamps_trainSN_mixed_valSN_real_resnet_custom_mini_normImage"

# --- Definición del Espacio de Búsqueda de Hiperparámetros (~12 Combinaciones) ---

# Tasas de Aprendizaje (Learning Rates): Probar el baseline y uno más bajo.
declare -a learning_rates=(
    "1e-3"
    "1e-4"
)

# Tasas de Dropout: El baseline es alto (0.7). Probemos valores moderados.
declare -a dropout_rates=(
    "0.4"
    "0.5"
    "0.6"
)

# --- Metadatos del Experimento ---
# Usa esto para agrupar todas las ejecuciones de esta búsqueda en MLflow.
exp_desc="rubin_gridsearch_quick_resnetmini_norm_image"
# El nombre de tu script de entrenamiento unificado.
TRAINING_SCRIPT="training_tf_custom.py" 

# --- Bucle Principal de la Búsqueda (Grid Search) ---
echo "Iniciando Búsqueda de Hiperparámetros para Rubin..."
echo "Usando la configuración base: ${HYDRA_CONFIG_PATH}/${HYDRA_CONFIG_NAME}.yaml"

total_runs=$(( ${#learning_rates[@]} * ${#arch_list[@]} * ${#dropout_rates[@]} ))
current_run=0

for lr in "${learning_rates[@]}"; do
  for dropout in "${dropout_rates[@]}"; do
    
    current_run=$((current_run + 1))
    echo "======================================================================"
    echo "EJECUCIÓN ${current_run} / ${total_runs}"
    echo "PARÁMETROS: LR=${lr}, Arch=${arch}, Dropout=${dropout}"
    echo "======================================================================"

    # Ejecutar el script de entrenamiento con los overrides de Hydra.
    CUDA_VISIBLE_DEVICES="MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b" python ${TRAINING_SCRIPT} \
      stamp_classifier.is_searching_hyperparameters=true \
      stamp_classifier.exp_description="${exp_desc}" \
      stamp_classifier.training.lr=$lr \
      stamp_classifier.loader.stamp_norm_type='image' \
      stamp_classifier.arch.dropout_rate=$dropout \
      hydra.run.dir="outputs/${exp_desc}/run_${current_run}" # Organizar logs de Hydra.

  done
done

echo "Búsqueda de hiperparámetros completada."