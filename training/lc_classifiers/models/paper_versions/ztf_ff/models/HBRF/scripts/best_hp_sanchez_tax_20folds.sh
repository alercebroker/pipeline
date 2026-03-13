#!/bin/bash

# Activar el entorno virtual
source ~/miniconda3/bin/activate hbrf_local

# Nombre base de la versión de datos
DATA_VERSION="250408_ndetge8"

echo "Entrenando modelo final (Sanchez Taxonomy - 20 folds)..."

# Generar lista de 0 a 19
FOLDS=$(python -c "print(list(range(20)))")

python training.py \
    --path_partition "../../data/partitions/${DATA_VERSION}_sanchez_tax_20folds/partitions.parquet" \
    --folds_to_run "$FOLDS" \
    --n_trees "[500]" \
    --criterion "['gini']" \
    --max_depth "[100]" \
    --num_workers 20

echo "Entrenamiento Sanchez Tax finalizado."