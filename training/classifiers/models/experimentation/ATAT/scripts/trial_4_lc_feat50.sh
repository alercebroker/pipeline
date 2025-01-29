# Activar el entorno virtual
source ~/miniconda3/bin/activate ATAT_ALeRCE

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_md_feat_mta \
--data_root_general data/ztf_forced_photometry/processed/dataset_pre241209_pos241222 \
--num_encoders 3 \
--embedding_size 192 \
--embedding_size_sub 384 \
--num_encoders_tab 3 \
--embedding_size_tab 128 \
--embedding_size_tab_sub 256 \
--lr_general 1e-4 \
--list_folds_general "[0]" \
--is_searching_hyperparameters_general \
--top_n_feats_general 50