### Stamp Classifier - Entrenamiento y Evaluación

Este proyecto contiene el pipeline de procesamiento y entrenamiento para un clasificador de recortes astronómicos (stamps).

---

### 📁 Preparación de los datos

Primero, se deben obtener los datos desde el servidor remoto ejecutando:

```bash
scp -r quimal_gpu:/storage/multilevel_stamp_classifier_data_wo_carrasco_davis \
/home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/data_acquisition/data/processed/consolidated_dataset.pkl
```

Luego, ubícate en el directorio de experimentación:

```bash
cd /home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/models/experimentation/stamp_full
```

Ejecuta el script de procesamiento:

```bash
python data_processor.py
```

Este paso genera dos archivos en la carpeta `data/`, ya divididos en conjuntos de entrenamiento, validación y test:

* `consolidated_ndarrays.pkl` (paso previo al normalizado)
* `normalized_ndarrays.pkl` (archivo final que se usa para entrenar)

---

### 📉 Entrenamiento del modelo

Utilizamos el archivo `normalized_ndarrays.pkl` para entrenar el modelo:

```bash
python train_and_save_best_models.py
```

Este proceso también genera logs compatibles con TensorBoard, donde se pueden visualizar las curvas de aprendizaje en términos de:

* F1-score
* Loss
* Accuracy

Para visualizar los resultados:

```bash
tensorboard --logdir=run_0/logs --port=6006
```

Y si estás en un servidor remoto, puedes hacer un reenvío de puerto (port forwarding) desde tu máquina local:

```bash
ssh -N -L 6006:localhost:6006 dmoreno@quimal-gpu
```

Abre luego en tu navegador:

```
http://localhost:6006
```

---

### 🔢 Evaluación del modelo

Para evaluar el modelo y generar la matriz de confusión, se puede usar el notebook:

```
data/inference.ipynb
```

> ⚠️ **Nota:** El notebook no está completamente ordenado, por lo que se recomienda revisarlo con atención para entender qué hace cada celda.

---

Para dudas o mejoras, por favor revisar el código y documentación correspondiente.
