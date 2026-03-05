#!/bin/bash
set -e 

# --- Configuración ---
export HYDRA_CONFIG_PATH="./configs"
export HYDRA_CONFIG_NAME="cnn_config_v1_real_rubin_allstamps_trainSN_valSN_real_resnet"

# Nombre del experimento para identificarlo en MLflow/Carpetas
exp_desc="rubin_resnet_frozen" 
TRAINING_SCRIPT="training_tf_custom.py"

# !!! GPU 1 !!!
GPU_ID="MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf" # Usa tu ID de GPU principal aquí

echo "Iniciando ResNet - MODO: TRANSFER LEARNING (Frozen)"
echo "Config base: ${HYDRA_CONFIG_NAME}"

# --- Parámetros Fijos para este script ---
ft_at="0"  # Backbone congelado
resize_dims="[64,64]"
dropout="0.5"

# --- Grid de Búsqueda ---
# LRs altos porque solo entrenamos el Head
declare -a learning_rates=("1e-3" "1e-4") 
declare -a dense_sizes=("256" "512")

run_counter=0

for lr in "${learning_rates[@]}"; do
    for d_size in "${dense_sizes[@]}"; do
        
        run_counter=$((run_counter + 1))
        
        echo "======================================================================"
        echo "[FROZEN RUN ${run_counter}] FT=${ft_at} | LR=${lr} | Dense=${d_size}"
        echo "======================================================================"

        dense_config="[{units: $d_size, activation: relu}]"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python ${TRAINING_SCRIPT} \
            stamp_classifier.is_searching_hyperparameters=true \
            stamp_classifier.exp_description="${exp_desc}" \
            stamp_classifier.arch.model_type="resnet" \
            stamp_classifier.arch.fine_tune_at=$ft_at \
            stamp_classifier.arch.resize_target="${resize_dims}" \
            stamp_classifier.arch.dense_config="${dense_config}" \
            stamp_classifier.arch.dropout_rate=$dropout \
            stamp_classifier.training.lr=$lr \
            hydra.run.dir="outputs/${exp_desc}/run_${run_counter}_ft${ft_at}_lr${lr}"
    
    done
done

echo ""
echo ">>> Runs de Transfer Learning finalizados."