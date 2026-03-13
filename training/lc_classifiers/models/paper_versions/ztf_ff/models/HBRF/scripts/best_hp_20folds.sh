#!/bin/bash

# Activar el entorno virtual
source ~/miniconda3/bin/activate hbrf_local

# Nombre base de la versión de datos
DATA_VERSION="250408_ndetge8"

echo "Entrenando modelo final (20 folds estándar)..."

# Nota: folds_to_run usa range(20) representado como lista
FOLDS=$(python -c "print(list(range(20)))")

python training.py \
    --path_partition "../../data/partitions/${DATA_VERSION}_20folds/partitions.parquet" \
    --folds_to_run "$FOLDS" \
    --n_trees "[500]" \
    --criterion "['gini']" \
    --max_depth "[100]" \
    --num_workers 20

echo "Entrenamiento finalizado."