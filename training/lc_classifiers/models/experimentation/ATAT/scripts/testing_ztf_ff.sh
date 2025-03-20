# Activar el entorno virtual
source ~/miniconda3/bin/activate mbappe_local

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_mta \
--name_dataset_general ztf_ff \
--data_root_general data/ztf_forced_photometry/processed/ds_pre241209_pos250120_detff_ndetge8 \
--num_encoders 3 \
--embedding_size 192 \
--embedding_size_sub 384 \
--num_encoders_tab 3 \
--embedding_size_tab 128 \
--embedding_size_tab_sub 256 \
--lr_general 1e-4 \
--use_lightcurves_err_general \
--list_folds_general "[0]"