# Alert Classification for the ALeRCE Broker System: The Forced Photometry Classifiers

This repository contains the training pipeline, evaluation notebooks, and production results for the ALeRCE light curve classifiers trained on forced photometry ZTF data. Two models are compared:

- **Mbappe** — [ATAT: Astronomical Transformer for time series and Tabular data](https://www.aanda.org/articles/aa/full_html/2024/09/aa49475-24/aa49475-24.html)
- **Squidward** — [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1)

---

## Project Structure

```
ztf_ff/
├── download_data.sh           # Script to download data and partitions
├── schema.py                  # Feature schema available in production
├── data/
│   ├── preprocessed/          # Light curve and feature data (.pkl files)
│   ├── partitions/            # K-fold partition files (.parquet)
│   └── scripts/
│       ├── data_partitioner.py          # Generates 5-fold partitions
│       └── get_partition_keep_test.py   # Expands to 20 folds, fixed test set
├── models/
│   ├── ATAT/                  # Mbappe model (see its own README)
│   └── HBRF/                  # Squidward model (see its own README)
├── notebooks/                 # Evaluation notebooks (tables and figures)
├── images/                    # Confusion matrices and evaluation time plots
├── results_prod/              # Production predictions on unlabeled data
```

---

## Data Acquisition

Download the data using the provided script:

```bash
bash download_data.sh data
```

This will download the following file:

- `objects_250410.parquet` — object catalog

And the following folders containing light curve and feature data:

```
data/preprocessed/data_250408_ndetge8_ao
data/preprocessed/data_250408_ndetge8_ao_shorten_features
```

These folders contain pickle files with dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py). Each dictionary includes the following keys: `'metadata'`, `'detections'`, `'non_detections'`, `'forced_photometry'`, `'xmatch'`, `'stamps'`, `'features'`, and `'predictions'`.

The data is split into chunks stored across multiple pickle files, in two variants:

1. **Without features** — contains light curve data (`'detections'`, `'non_detections'`, `'forced_photometry'`). Example: `data_250408_ndetge8_ao/astro_objects_batch_000.pkl`

2. **With features** — contains `'metadata'` and `'features'`. The number of days used to compute features is indicated at the start of the filename. Example: `data_250408_ndetge8_ao_shorten_features/{days}_astro_objects_batch_000.pkl`. These files also include light curve data, but note that the **Modified Julian Date (MJD)** has been modified during feature computation.

---

## Partitions

Download the partitions used for Mbappe and Squidward directly:

```bash
bash download_data.sh partitions
```

Alternatively, generate new partitions from scratch. First, create a 5-fold split over the new ALeRCE taxonomy (21 classes) (used for hyperparameter search):

```bash
python data/scripts/data_partitioner.py
```

Then expand to 20 folds for both the new ALeRCE taxonomy and the [Sánchez et al. taxonomy](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1), keeping the test set fixed:

```bash
python data/scripts/get_partition_keep_test.py
```

Partitions are stored as `partitions.parquet` files with the following columns:

| Column | Description |
|---|---|
| `oid` | Object ID |
| `class_name` | Object label |
| `ra`, `dec` | Right Ascension and Declination |
| `partition` | Data split: `test`, `training_i`, or `validation_i` |

K-fold cross-validation is used to ensure consistent train/validation/test splits across folds. Objects labeled `test` are never used for training or validation.

---

## Models

Each model has its own directory under `models/` with a dedicated README explaining how to run training and inference:

- `models/ATAT/` — **Mbappe** (ATAT transformer)
- `models/HBRF/` — **Squidward** (Hierarchical Balanced Random Forest)

---

## Notebooks

The `notebooks/` folder contains all evaluation results presented in the paper, including confusion matrices and timing analyses for both models and taxonomies.

---

## Production Results

`results_prod/` contains model predictions on unlabeled production data — objects drawn from the ALeRCE database and classified by each model. Results are available for two production runs:

- `objs_rand350000_2025-12-23_*` — 350k objects, December 2025
- `objs_rand500000_2025-03-06_*` — 500k objects, March 2025 (pending)

Each run has three variants: `_features`, `_mbappe`, and `_squidward`, also available as `.tar.gz` archives.

---

## Additional Files

- **`schema.py`** — defines the feature schema available in production, useful for aligning training features with deployed pipelines.
- **`images/`** — confusion matrices (`cm_*`) and evaluation time plots (`eval_time_*`) for both models and taxonomies, in PDF format.