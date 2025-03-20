# Activar el entorno virtual
source ~/miniconda3/bin/activate mbappe_local

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_md_feat_mta \
--experiment_name_general trial_0_sanchez_tax \
--name_dataset_general ztf_ff_sanchez_tax \
--data_root_general data/ztf_forced_photometry/processed/ds_pre241209_pos250218_detff_ndetge8_sanchez_tax \
--num_encoders 2 \
--embedding_size 96 \
--embedding_size_sub 192 \
--num_encoders_tab 2 \
--embedding_size_tab 64 \
--embedding_size_tab_sub 128 \
--lr_general 1e-4 \
--list_folds_general "[1]"