# Data Preprocessing Overview

This document provides an overview of the preprocessing steps applied to the **ZTF forced photometry** data. It includes instructions for obtaining raw data and preprocessing it to generate the **AstroObjects**, which are used as input for the models.

## ZTF Forced Photometry

You have two options for data preparation:

1. **Get the raw data:** Perform preprocessing from scratch to generate the preprocessed data used by the **Mbappe** and **Squidward** models, or apply your own preprocessing methods.

2. **Get the preprocessed data:** Directly download the data that was used to train the **Mbappe** and **Squidward** models.

### 1. Get the Raw data [WORK IN PROCESS]

To obtain the raw data, first create a directory for it:

```
mkdir raw
```

The raw `ztf_forced_photometry` data is stored on the server `quimal-gpu` at the following location: `/storage/ztf_forced_photometry`.

To copy the data to your local machine, run the following commands:

```
scp -r <user_name>@146.83.185.167:/storage/ztf_forced_photometry/detections.parquet ./raw
scp -r <user_name>@146.83.185.167:/storage/ztf_forced_photometry/objects.parquet ./raw
```

To preprocess the raw data from scratch using the same methods as **Mbappe** and **Squidward** run:

```
python scripts/preprocessing.py
```

**Preprocessing details: (WORK IN PROCESS, so start in the step 2. for now)**

[ESCRIBIR AQUÍ LA EXPLICACIÓN DE QUE SE HACE EXACTAMENTE.] Algunas notas que serviran para esto:

Processing (Get LC AstroObjects and AstroObjects with computed features).

2. Obtener la tabla processed (la hago yo en mi notebook).

2.1 Alejandra toma la tabla processed, las convierte a parquet files y luego a pickles files que contienen diccionario de los AstroObjects.

2.1.1 Aqui se dejan objetos con Minimo 2 Detecciones.

2.1.2 Se eliminan some spurious microlensing events.
* **microlensing events to be removed**. Some spurious microlensing events were detected by the ALeRCE team. These are stored in csv files that only contain the object IDs of objects to be removed/kept.

2.1.3 Se elimina los objetos de clase ZZ.

2.1.4 Preguntar a Ale si hay algo más.

2.2 Computar features (para diferentes largos de la curva de luz si es necesario).
---> preprocessing.py
* from data_241209_ao ---> data_241209_ao_shorten_features


You can also perform your own preprocessing if desired.


### 2. Get the Preprocessed Data

If you prefer to use the preprocessed data directly, first create a directory for it:

```
mkdir preprocessed
```

The preprocessed `ztf_forced_photometry` data is stored on the server `quimal-gpu` at `/storage/ztf_forced_photometry`.

To copy the preprocessed data to your local machine, use the following commands:

```
scp -r <user_name>@146.83.185.167:/storage/ztf_forced_photometry/data_241209_ao ./preprocessed/data_241209_ao
scp -r <user_name>@146.83.185.167:/storage/ztf_forced_photometry/data_241209_ao_features ./preprocessed/data_241209_ao_shorten_features
```

These folders contain pickle files with dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py). Each dictionary includes the following keys: `'metadata'`, `'detections'`, `'non_detections'`, `'forced_photometry'`, `'xmatch'`, `'stamps'`, `'features'`, and `'predictions'`. 

The data is divided into chunks and stored across multiple pickle files. There are two types of files:

1. **Without features**: These files contain the dataset with light curves, i.e., `'detections'`, `'non_detections'`, and `'forced_photometry'` information. For example: `data_241209_ao/astro_objects_batch_000.pkl`

2. **With features**: These files contain the `'metadata'`, and `'features'` information. The number of days used to calculate the features is indicated at the beginning of the file name. For example: `data_241209_ao_shorten_features/{days}_astro_objects_batch_000.pkl`. These files also include light curve data; however, the **Modified Julian Date (MJD)** has been altered during the feature calculation process.

### Generate Partitions

Create a directory for the partitions:

To create partitions, start by creating a directory:

Partitions are generated using the **Without Features** dataset, i.e., `data_241209_ao`. To obtain the partitions used for **Mbappe** and **Squidward**, run:

```
scp -r <user_name>@146.83.185.167:/storage/ztf_forced_photometry/partitions/241209 ./preprocessed/partitions
```

Alternatively, you can create new partitions using the provided script:

```
python scripts/data_partitioner.py
```

* **parquet files containing partitions**. We use K-fold cross-validation to ensure consistency across train-validation-test sets. A parquet file named partitions.parquet contains the following columns:
    * `'oid'`: Object ID
    * `'alerceclass'`: Object label
    * `'ra', 'dec'`: Right Ascension and Declination
    * `'partition'`: Indicates the data split, which can be one of the following`:
        * `'test'`: The object is never used for training or validation.
        * `'training_i'`: The object is used for training in the i-th fold.
        * `'validation_i'`: The object is used for validation in the i-th fold.


## Notes

For additional support, refer to the documentation or contact the ALeRCE team.
