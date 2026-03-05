# Entrenamiento del Stamp Classifier

## Versionamiento de datos (DVC)

Idealmente, DVC debería utilizarse para versionar los datasets y las particiones del dataset utilizadas durante el entrenamiento.

Además, DVC debería generar algún archivo de metadatos (por ejemplo `meta.yaml`) que permita posteriormente enlazar el modelo entrenado con la versión del dataset y la partición utilizada.

De esta manera, MLflow podría guardar esta información, permitiendo rastrear exactamente con qué datos y particiones se entrenó cada modelo.

> Actualmente esta integración no está implementada completamente. Por mientras, hacer lo que dice abajo.

---

## Obtención de los datos

Para obtener el dataset, debes crear un enlace simbólico al dataset de Rubin:
```bash
ln -s /home/rubin_dp1/datasets/lsst/ts_stamps_v0.0.2_comm_4candmax \
/home/<quimal_user>/pipeline/training/stamp_classifier/data_acquisition/rubin/data/processed
```

---

## Particiones

Una vez obtenidos los datos, debes obtener o generar las particiones del dataset.

### Usar las particiones existentes

Para utilizar las particiones usadas en Stamp versión 2.0.0 y 2.0.1, crea el siguiente enlace simbólico:
```bash
ln -s /home/rubin_dp1/datasets/lsst/ts_stamps_v0.0.2_comm_4candmax/partitions/partitions_trainSN_mixed_valSN_real_eval_firstStamp_v1 \
/home/<quimal_user>/pipeline/training/stamp_classifier/data_acquisition/rubin/data/processed/partitions
```

### Generación de particiones

La generación de particiones idealmente debería hacerse manualmente, ya que el número de supernovas en el dataset es bajo y se busca que el conjunto de test sea representativo para esta clase.

**Recomendaciones para generar las particiones:**

1. Separar primero las supernovas en el conjunto de test y luego hacer el split considerando todas las clases. 
   Dado que hay pocas supernovas, es importante que estén bien representadas en el test.

2. Mantener objetos del mismo objeto TNS dentro del mismo subset  
   (idealmente dentro del conjunto de entrenamiento).

3. Mantener objetos con distancia menor a 0.2 dentro del mismo subset  
   (idealmente dentro del conjunto de entrenamiento).

4. Agregar los datos simulados dentro de los subsets  
   (train y/o validation).

### Herramientas disponibles

Hay dos opciones disponibles para generar las particiones:

- **Script automático de particiones:** Permite generar particiones automáticamente, pero no considera las restricciones mencionadas arriba.
- **Notebook de ejemplo para particiones manuales:** Existe un notebook que muestra cómo generar las particiones manualmente, aunque actualmente no está bien organizado.

> ⚠️ **Importante:** Las condiciones 2 y 3 son importantes para evitar *data leakage*.

---

## Entrenamiento del modelo

Una vez que los datos y las particiones están en el lugar correcto, se puede entrenar el modelo usando:
```bash
python training_tf_custom.py \
--cnn_config_v1_real_rubin_allstamps_trainSN_mixed_valSN_real_modified_normIgnacio
```

El argumento corresponde a una configuración de Hydra (archivo YAML) ubicada en la carpeta `configs`.

Este comando entrenará el modelo utilizando los hiperparámetros por defecto definidos en la configuración.

---

## Búsqueda de hiperparámetros

Para realizar búsqueda de hiperparámetros, lo más recomendable es utilizar los scripts `.sh`. Por ejemplo:
```bash
bash scripts/search_stamp_rubin_real_v1_architecture_allstamps_trainSN_mixed_valSN_real_modified.sh
```

Este script llama a una `config.yaml` de la carpeta `configs` y llama al script `training_tf_custom.py`. Existen otros scripts de ejemplo que puedes revisar o modificar según lo necesites:

- `run_resnet_finetune.sh`
- `run_resnet_frozen.sh`
- `search_stamp_rubin_real_v1_architecture_allstamps_fforster_mixed_modified_multiresolucion.sh`
- `search_stamp_rubin_real_v1_architecture_allstamps_trainSN_mixed_valSN_real_modified_band.sh`
- `search_stamp_rubin_real_v1_architecture_allstamps_trainSN_mixed_valSN_real_modified_f05.sh`
- `search_stamp_rubin_real_v1_architecture_allstamps_trainSN_mixed_valSN_real_modified_normImage.sh`

> Algunos de estos scripts pueden contener argumentos obsoletos o en desuso, por lo que se recomienda revisarlos antes de utilizarlos.

---

## Cosas hardcodeadas que deberían corregirse

Actualmente existen algunos elementos hardcodeados en el código que deberían mejorarse en el futuro.

### Configuración de metadatos

Actualmente, si se configura por ejemplo:
```yaml
use_metadata: true
use_coords: false
```

no necesariamente funciona como se esperaría. La configuración de metadatos debería refactorizarse para que su comportamiento sea consistente.

### Nombre de columnas en entrenamiento y producción

Acutlamente entrenamos con columnas llamadas:
- oid
- flux_Science_data
- flux_Difference_data
- flux_Template_data

Las cuales no matchean con los nombres que vienen en producción.
- diaObjectId
- visit_image
- 

Ver el tema de las columnas en la config de hparams: flux_Science_data —> visit_image, etc… oid —> diaObjectId y en el test_sample_data.pkl que se guarda

