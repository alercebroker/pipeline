# Activar el entorno virtual
source ~/miniconda3/bin/activate mbappe_local

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_md_feat_mta \
--experiment_name_general trial_6 \
--data_root_general data/ztf_forced_photometry/processed/ds_pre241209_pos250120_detff_ndetge8 \
--num_encoders 4 \
--embedding_size 384 \
--embedding_size_sub 768 \
--num_encoders_tab 2 \
--embedding_size_tab 64 \
--embedding_size_tab_sub 128 \
--lr_general 1e-4 \
--list_folds_general "[0, 1, 2, 3, 4]" \
--is_searching_hyperparameters_general