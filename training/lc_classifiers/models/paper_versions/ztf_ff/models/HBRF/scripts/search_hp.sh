#!/bin/bash

# Activar el entorno virtual
source ~/miniconda3/bin/activate hbrf_local

# Nombre base de la versión de datos
DATA_VERSION="250408_ndetge8"

echo "Iniciando Búsqueda de Hiperparámetros..."

python training.py \
    --path_partition "../../data/partitions/${DATA_VERSION}/partitions.parquet" \
    --searching_hyperparameters \
    --folds_to_run "[0, 1, 2, 3, 4]" \
    --n_trees "[10, 100, 350, 500]" \
    --criterion "['gini', 'entropy']" \
    --max_depth "[10, 100, None]" \
    --num_workers 20

echo "Búsqueda finalizada."