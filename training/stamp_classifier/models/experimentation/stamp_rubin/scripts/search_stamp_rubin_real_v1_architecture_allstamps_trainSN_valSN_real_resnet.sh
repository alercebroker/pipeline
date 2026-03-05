#!/bin/bash
set -e # Salir si hay errores

# --- Configuración ---
export HYDRA_CONFIG_PATH="./configs"
export HYDRA_CONFIG_NAME="cnn_config_v1_real_rubin_allstamps_trainSN_valSN_real_resnet"

exp_desc="rubin_resnet_optimization"
TRAINING_SCRIPT="training_tf_custom.py"
GPU_ID="MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf" # Tu ID de GPU

echo "Iniciando Búsqueda de Hiperparámetros EXCLUSIVA para ResNet..."
echo "Config base: ${HYDRA_CONFIG_NAME}"

# 1. Estrategias de Fine Tuning
#    "0"  = Transfer Learning (Backbone congelado, solo entrena el head).
#    "48" = Fine Tuning (Descongela el último bloque de ResNet aprox).
declare -a ft_strategies=("0" "48")

# 2. Tamaño del 'Head' (Capas densas finales)
#    Probamos si necesita muchas neuronas o pocas para adaptar los features de ResNet.
declare -a dense_sizes=("256" "512")

# 3. Dropout (fijo o variable, aquí dejamos uno estándar)
dropout="0.5"

# 4. Tamaño de imagen (Resize)
#    [64,64] es un buen balance. [32,32] es muy chico para ResNet, [224,224] muy lento.
resize_dims="[64,64]"

run_counter=0

for ft_at in "${ft_strategies[@]}"; do
    
    # --- LÓGICA DE LEARNING RATE ADAPTATIVO ---
    if [ "$ft_at" -eq "0" ]; then
        # BACKBONE CONGELADO: Podemos ser agresivos con el LR.
        # Probamos 1e-3 (estándar) y 1e-4 (más preciso).
        current_lrs=("1e-3" "1e-4")
        mode_name="TransferLearning"
    else
        # FINE TUNING: Debemos ser conservadores.
        # Probamos 1e-5 y 1e-6 para no romper los pesos de ImageNet.
        current_lrs=("1e-5" "1e-6")
        mode_name="FineTuning"
    fi

    for lr in "${current_lrs[@]}"; do
        for d_size in "${dense_sizes[@]}"; do
            
            run_counter=$((run_counter + 1))
            
            echo "======================================================================"
            echo "[RUN ${run_counter}] Mode: ${mode_name} (FT=${ft_at}) | LR=${lr} | Dense=${d_size}"
            echo "======================================================================"

            # Construir config de la capa densa
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
done

echo ""
echo ">>> Búsqueda de ResNet finalizada. Total de ejecuciones: ${run_counter}"