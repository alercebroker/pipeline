# Activar el entorno virtual
source ~/miniconda3/bin/activate mbappe_local

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_mta \
--name_dataset_general atlas \
--data_root_general data/atlas/processed/ds_pre241209_pos250304_detff_limitMJD_woFluxMask_detected \
--num_encoders 3 \
--embedding_size 192 \
--embedding_size_sub 384 \
--num_encoders_tab 3 \
--embedding_size_tab 128 \
--embedding_size_tab_sub 256 \
--lr_general 1e-4 \
--use_lightcurves_err_general \
--list_folds_general "[0]"