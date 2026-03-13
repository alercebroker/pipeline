# ATAT — Astronomical Transformer for time series And Tabular data
 
The model called mbappe in production proposed in [ATAT: Astronomical Transformer for time series and Tabular data](https://www.aanda.org/articles/aa/full_html/2024/09/aa49475-24/aa49475-24.html) was now applied and adapted over forced photometric ZTF data. We use [MLflow](https://mlflow.org/) for experiment tracking.

---

## Prerequisites

Before running ATAT, make sure the following ZTF datasets are available at the expected paths relative to the project root:

**Light curve and feature data:**
```
../../data/preprocessed/data_250408_ndetge8_ao
../../data/preprocessed/data_250408_ndetge8_ao_shorten_features
```

**Partition files:**
```
../../data/partitions/250408_ndetge8/partitions.parquet
../../data/partitions/250408_ndetge8_20folds/partitions.parquet
../../data/partitions/250408_ndetge8_sanchez_tax_20folds/partitions.parquet
```

---
 
## Installation
 
Install the required Python dependencies from the project root:
 
```bash
pip install -r requirements.txt
```

---

## Setup & Data Processing

From the `ATAT/` directory, run the data processor to generate the `.h5` input files required by the model:

```bash
python data_processor.py
```

This step must be completed before running any training or inference scripts.

---

## Training

Training scripts are organized around two taxonomies and are named accordingly:

| Script prefix | Taxonomy |
|---|---|
| `best_hp_*` | **New ALeRCE taxonomy** (21 classes) |
| `best_hp_sanchez_tax_*` | **Sánchez et al. taxonomy** — the original ALeRCE taxonomy described in [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1) |

These scripts use the best hyperparameter configuration found during the search performed by the `search_hp_*` scripts.

Training is split across multiple scripts, each covering 3 folds at a time (up to 20 folds total). To run the full 20-fold training for the **new taxonomy**:

```bash
bash scripts/best_hp_20folds_0.sh
bash scripts/best_hp_20folds_1.sh
# ...
bash scripts/best_hp_20folds_6.sh
```

The same approach applies to the **Sánchez taxonomy** — just replace the script prefix with `best_hp_sanchez_tax_`.

---

## Inference

### Standard inference

Inference is automatically triggered at the end of each training script via a call to `inference.py` — no separate step is needed for standard evaluation.

### Inference by number of days after first alert

To evaluate model performance as a function of how many days of data are available after the first alert, run:

```bash
python inference_ndays.py
```

Before running, configure the dataset name inside the script. This name must match the corresponding MLflow experiment name. The script will load all model runs from MLflow and perform inference over light curves of varying lengths, producing predictions across different temporal windows.

> **Note:** If training was only completed for one taxonomy, configure `inference_ndays.py` to match that experiment name before running.

---

## Experiment Tracking (MLflow)

All training runs are tracked with MLflow. To launch the MLflow UI and browse experiments, models, and metrics:

```bash
mlflow ui --backend-store-uri=file:./results/ml-runs --port 7800
```

Then open [http://localhost:7800](http://localhost:7800) in your browser.

---

## Project Structure

```
ATAT/
├── calibration.py             # Model calibration utilities
├── custom_parser.py           # Argument parsing
├── data_processor.py          # Generates .h5 input files from raw data
├── inference.py               # Called internally during training
├── inference_ndays.py         # Inference over varying light curve lengths
├── training.py                # Main training entry point
├── utils.py                   # Shared utilities
├── requirements.txt
├── configs/
│   └── training.yaml          # Training configuration
├── data/
│   └── processed/             # Output of data_processor.py (.h5 files)
├── scripts/
│   ├── best_hp_20folds_0.sh           # New taxonomy training (20 folds, 3 folds each)
│   ├── best_hp_20folds_[1-6].sh
│   ├── best_hp_sanchez_tax_20folds_0.sh   # Sánchez taxonomy training
│   ├── best_hp_sanchez_tax_20folds_[1-6].sh
│   └── search_hp_trial_[0-17].sh          # Hyperparameter search scripts
├── src/
│   ├── data/                  # Data loading and preprocessing modules
│   ├── layers/                # Custom neural network layers
│   ├── models/                # Model definitions
│   ├── training/              # Training loops and callbacks
│   └── utils/                 # Internal utilities
├── results/
│   └── ml-runs/               # Local MLflow experiment store
└── utils/
    ├── clean_mlflow.py        # MLflow run cleanup script
    └── merge_mlflow.py        # Merge MLflow experiment stores
```




