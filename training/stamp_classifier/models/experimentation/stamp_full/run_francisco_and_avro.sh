#!/bin/bash

# Valores por defecto
stamp="full"
bn=True
crop=False
use_metadata=False
focal_loss=False

# Path_data que quieres probar
path_datas=(
"./data/normalized_ndarrays_2025-06-06_04-04.pkl"
"./data/normalized_ndarrays_hasavro_2025-06-06_04-26.pkl"
)

# Arreglo con los nombres de los parámetros y sus valores no default
params=("stamp" "bn" "crop" "use_metadata" "focal_loss")
defaults=("full" True False False False)
non_defaults=("modified" False True True True)

# Total de parámetros
N=${#params[@]}

# Recorre cada path_data
for path_data in "${path_datas[@]}"; do
    echo "Running with path_data: $path_data"

    # Siempre recorrer de manera incremental
    for i in $(seq 0 $N); do
        args=("${defaults[@]}")

        # Cambiar los primeros i argumentos al valor no default
        for ((j=0; j<i; j++)); do
            args[$j]=${non_defaults[$j]}
        done

        # Construir comando
        echo "Running with: stamp=${args[0]} bn=${args[1]} crop=${args[2]} use_metadata=${args[3]} focal_loss=${args[4]} path_data=$path_data"

        python train_and_save_best_models.py \
        --stamp "${args[0]}" \
        --bn "${args[1]}" \
        --crop "${args[2]}" \
        --use_metadata "${args[3]}" \
        --focal_loss "${args[4]}" \
        --path_data "$path_data"
    done
done
