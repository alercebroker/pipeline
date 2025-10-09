#!/bin/bash

# Define lista de arquitecturas a evaluar (cada una como una combinación de filtros)
declare -a filters_list=(
    "16"
    "16-32"
    "32-64"
    "32-64-128"
    "32-64-128-256"
)

# Define configuraciones de capas densas
declare -a dense_units_list=(
    "32"
    "64"
    "128"
    "256"
)

# Define valores de dropout a probar
declare -a dropout_list=(
    0.0
    0.2
    0.5
    0.7
)


exp_desc="rubinAGNvsVS"

# Loop por todas las combinaciones
for filters in "${filters_list[@]}"; do
  for dense_units in "${dense_units_list[@]}"; do
    for dropout in "${dropout_list[@]}"; do

        # Crear configuración YAML directamente para conv_config
        conv_config="["
        IFS='-' read -ra filter_array <<< "$filters"
        for f in "${filter_array[@]}"; do
        conv_config+="{filters: $f, kernel_size: [3, 3], activation: relu, pool: true, pool_size: [2, 2]}, "
        done
        conv_config="${conv_config%, }]"  # Quitar coma final

        # Ejecutar entrenamiento
        python training_custom.py \
        stamp_classifier.is_searching_hyperparameters=true \
        stamp_classifier.arch.conv_config="${conv_config}" \
        stamp_classifier.arch.dense_config="[ {units: $dense_units, activation: tanh} ]" \
        stamp_classifier.arch.dropout_rate=$dropout \
        stamp_classifier.exp_description="${exp_desc}"

    done
  done
done
