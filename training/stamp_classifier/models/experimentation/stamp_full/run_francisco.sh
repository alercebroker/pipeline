#!/bin/bash

# Valores por defecto
stamp="full"
bn=True
crop=False
use_metadata=False
use_only_avro=False
focal_loss=False

# Arreglo con los nombres de los parámetros y sus valores no default
params=("stamp" "bn" "crop" "use_metadata" "use_only_avro" "focal_loss")
defaults=("full" True False False False False)
non_defaults=("modified" False True True False True)

# Total de parámetros
N=${#params[@]}

# Recorre combinaciones incrementales
for i in $(seq 0 $N); do
	# Copiar los valores por defecto
	args=("${defaults[@]}")
	# Cambiar los primeros i argumentos al valor no default
	for ((j=0; j<i; j++)); do
		args[$j]=${non_defaults[$j]}
		done
	# Construir comando
	echo "Running with: stamp=${args[0]} bn=${args[1]} crop=${args[2]} metadata=${args[3]} avro=${args[4]} focal_loss=${args[5]}"
	python train_and_save_best_models.py \
	--stamp "${args[0]}" \
	--bn "${args[1]}" \
	--crop "${args[2]}" 
	--use_metadata "${args[3]}" \
	--use_only_avro "${args[4]}" \
	--focal_loss "${args[5]}"
	done