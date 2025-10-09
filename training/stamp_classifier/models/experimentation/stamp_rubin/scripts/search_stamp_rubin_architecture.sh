#!/bin/bash
set -e # Salir inmediatamente si un comando falla.

# --- Configuración de Hydra ---
# Exporta las variables de entorno para que Hydra encuentre el archivo de config base.
export HYDRA_CONFIG_PATH="./configs"
export HYDRA_CONFIG_NAME="cnn_config_rubin"

# --- Definición del Espacio de Búsqueda de Hiperparámetros (~12 Combinaciones) ---

# Tasas de Aprendizaje (Learning Rates): Probar el baseline y uno más bajo.
declare -a learning_rates=(
    "7e-4"
    "2e-4"
)

# Arquitecturas CNN: Probar modelos simples y progresivamente más complejos.
# La cadena representa los filtros de cada bloque convolucional.
declare -a arch_list=(
    "8-16"         # Un modelo similar al baseline, pero más simple.
    "16-32"        # Un modelo un poco más ancho.
    "8-16-32"      # Un modelo un poco más profundo.
)

# Tasas de Dropout: El baseline es alto (0.7). Probemos valores moderados.
declare -a dropout_rates=(
    "0.4"
    "0.6"
)

# --- Metadatos del Experimento ---
# Usa esto para agrupar todas las ejecuciones de esta búsqueda en MLflow.
exp_desc="rubin_gridsearch_quick"
# El nombre de tu script de entrenamiento unificado.
TRAINING_SCRIPT="training_tf_custom.py" 

# --- Bucle Principal de la Búsqueda (Grid Search) ---
echo "Iniciando Búsqueda de Hiperparámetros para Rubin..."
echo "Usando la configuración base: ${HYDRA_CONFIG_PATH}/${HYDRA_CONFIG_NAME}.yaml"

total_runs=$(( ${#learning_rates[@]} * ${#arch_list[@]} * ${#dropout_rates[@]} ))
current_run=0

for lr in "${learning_rates[@]}"; do
  for arch in "${arch_list[@]}"; do
    for dropout in "${dropout_rates[@]}"; do
      
      current_run=$((current_run + 1))
      echo "======================================================================"
      echo "EJECUCIÓN ${current_run} / ${total_runs}"
      echo "PARÁMETROS: LR=${lr}, Arch=${arch}, Dropout=${dropout}"
      echo "======================================================================"

      # Construir dinámicamente el string de 'conv_config' para Hydra.
      # Este patrón es simple y efectivo: un bloque Conv + Pool por cada número.
      conv_config="["
      IFS='-' read -ra filter_array <<< "$arch"
      for f in "${filter_array[@]}"; do
          # Usamos kernel_size [2,2] como en tu baseline.
          conv_config+="{filters: $f, kernel_size: [2, 2], activation: relu, pool: true, pool_size: [2, 2]}, "
      done
      conv_config="${conv_config%, }]" # Eliminar la última coma y espacio.

      # Ejecutar el script de entrenamiento con los overrides de Hydra.
      CUDA_VISIBLE_DEVICES="MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf" python ${TRAINING_SCRIPT} \
        stamp_classifier.is_searching_hyperparameters=true \
        stamp_classifier.exp_description="${exp_desc}" \
        stamp_classifier.training.lr=$lr \
        stamp_classifier.arch.conv_config="${conv_config}" \
        stamp_classifier.arch.dropout_rate=$dropout \
        hydra.run.dir="outputs/${exp_desc}/run_${current_run}" # Organizar logs de Hydra.

    done
  done
done

echo "Búsqueda de hiperparámetros completada."