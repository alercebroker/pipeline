#!/bin/bash
# Batch 1 - Full model with batchnorm, no crop, no focal loss

# Define variables
PATH_DATA="./data/normalized_ndarrays_hasavro_2025-06-06_04-26.pkl"
STAMP="full"
RESULTS_FOLDER="./results_v1"
GPU=MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf  # <- You select which GPU

coord_type="cartesian"
norm_type="none"

# Print what will run
echo "Running Batch 1:"
echo "PATH_DATA=$PATH_DATA"
echo "STAMP=$STAMP"
echo "RESULTS_FOLDER=$RESULTS_FOLDER"

# Run Python script
CUDA_VISIBLE_DEVICES=$GPU python train_and_save_best_models.py \
    --path_data "$PATH_DATA" \
    --results_folder "$RESULTS_FOLDER" \
    --stamp "$STAMP" \
    --coord_type "$coord_type" \
    --norm_type "$norm_type" \
    --bn