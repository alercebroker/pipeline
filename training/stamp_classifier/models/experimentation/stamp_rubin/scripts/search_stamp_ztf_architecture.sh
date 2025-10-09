#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Hydra Configuration ---
# Export the environment variables that your Python script will read.
# This tells Hydra where to find the base configuration file.
export HYDRA_CONFIG_PATH="./configs"      # The directory containing your YAML files
export HYDRA_CONFIG_NAME="cnn_config_ztf"  # The base config file name (without .yaml)

# --- Define Hyperparameter Search Space ---

# Learning Rates (Logarithmic scale is best)
declare -a learning_rates=(
    "5e-4"
    "1e-4"
)

# CNN Architectures (string represents filters in each block)
declare -a arch_list=(
    "32-64"
    "16-32-64"
    "32-64-128"
)

# Dense Layer Units
declare -a dense_units_list=(
    "64"
)

# Dropout Rates
declare -a dropout_rates=(
    "0.2"
    "0.5"
)

# --- Experiment Metadata ---
exp_desc="ztf_crop31_gridsearch_quick"
TRAINING_SCRIPT="training_tf_custom.py" # <--- IMPORTANT: RENAME THIS

# --- Main Loop for Grid Search ---
echo "Starting Quick Hyperparameter Search..."
echo "Using Hydra base config: ${HYDRA_CONFIG_PATH}/${HYDRA_CONFIG_NAME}.yaml"
total_runs=$(( ${#learning_rates[@]} * ${#arch_list[@]} * ${#dense_units_list[@]} * ${#dropout_rates[@]} ))
current_run=0

for lr in "${learning_rates[@]}"; do
  for arch in "${arch_list[@]}"; do
    for dense_units in "${dense_units_list[@]}"; do
      for dropout in "${dropout_rates[@]}"; do
        
        current_run=$((current_run + 1))
        echo "======================================================================"
        echo "RUN ${current_run} / ${total_runs}"
        echo "PARAMS: LR=${lr}, Arch=${arch}, Dense=${dense_units}, Dropout=${dropout}"
        echo "======================================================================"

        # Dynamically build the conv_config string for Hydra
        conv_config="["
        IFS='-' read -ra filter_array <<< "$arch"
        for f in "${filter_array[@]}"; do
            conv_config+="{filters: $f, kernel_size: [3, 3], activation: relu, pool: true, pool_size: [2, 2]}, "
        done
        conv_config="${conv_config%, }]"

        # Execute the training script with Hydra overrides
        python ${TRAINING_SCRIPT} \
          stamp_classifier.is_searching_hyperparameters=true \
          stamp_classifier.exp_description="${exp_desc}" \
          stamp_classifier.training.lr=$lr \
          stamp_classifier.arch.conv_config="${conv_config}" \
          stamp_classifier.arch.dense_config="[ {units: $dense_units, activation: tanh} ]" \
          stamp_classifier.arch.dropout_rate=$dropout \
          hydra.run.dir="outputs/${exp_desc}/run_${current_run}"

      done
    done
  done
done

echo "Hyperparameter search complete."