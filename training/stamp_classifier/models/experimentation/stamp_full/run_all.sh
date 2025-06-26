#!/bin/bash

# Recorre todos los archivos .sh dentro de bash/all_classes_v1
for script in bash/all_classes_v1/*.sh; do
    echo "Running $script..."
    bash "$script"
done

#bash bash/all_classes_v1/run_avro_bn_spherical_md_zs.sh
#bash bash/all_classes_v1/run_avro_bn_cartesian_md_qt.sh
#bash bash/all_classes_v1/run_avro_bn_cartesian_md_zs.sh
#bash bash/all_classes_v1/run_avro_bn_cartesian.sh
#bash bash/all_classes_v1/run_avro_bn_spherical_md_qt.sh
