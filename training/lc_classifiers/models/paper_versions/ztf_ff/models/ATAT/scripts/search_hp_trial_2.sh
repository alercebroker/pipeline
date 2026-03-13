# Activar el entorno virtual
source ~/miniconda3/bin/activate mbappe_local_v1

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_md_feat_mta \
--experiment_name_general trial_2 \
--name_dataset_general ztf_ff \
--data_root_general data/processed/ds_pre250408_ndetge8_pos250728 \
--num_encoders 2 \
--embedding_size 96 \
--embedding_size_sub 192 \
--num_encoders_tab 4 \
--embedding_size_tab 256 \
--embedding_size_tab_sub 512 \
--lr_general 1e-4 \
--list_folds_general "[0, 1, 2, 3, 4]" \
--is_searching_hyperparameters_general