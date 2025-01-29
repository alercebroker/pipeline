# Activar el entorno virtual
source ~/miniconda3/bin/activate ATAT_ALeRCE

# Exportar variables de entorno si es necesario
export CUDA_VISIBLE_DEVICES=MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b

# Ejecutar el script de entrenamiento
python training.py \
--experiment_type_general lc_md_feat_mta \
--experiment_name_general '0.3.4 only det' \
--data_root_general data/ztf_forced_photometry/processed/ds_pre241209_pos250120_detff_ndetge8 \
--num_encoders 3 \
--embedding_size 192 \
--embedding_size_sub 384 \
--num_encoders_tab 3 \
--embedding_size_tab 128 \
--embedding_size_tab_sub 256 \
--lr_general 1e-4 \
--list_folds_general "[0]" \
--use_only_det_general