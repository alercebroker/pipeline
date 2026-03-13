# Alert Classification for the ALeRCE Broker System: The Light Curve Classifier

This classifier, known as **Squidward** in production, was originally proposed in [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1) and has now been applied and adapted to forced photometry ZTF data.

---

## Prerequisites

Before running Squidward, make sure the following datasets are available at the expected paths relative to the project root:

**Feature data:**
```
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

## Training

Training scripts are organized around two taxonomies:

| Script | Taxonomy |
|---|---|
| `best_hp_20folds.sh` | **New ALeRCE taxonomy** (21 classes) |
| `best_hp_sanchez_tax_20folds.sh` | **Sánchez et al. taxonomy** — the original ALeRCE taxonomy described in [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://iopscience.iop.org/article/10.3847/1538-3881/abd5c1) |

Both scripts use the best hyperparameter configuration found via `search_hp.sh`. Unlike ATAT, each taxonomy runs in a single script covering all 20 folds at once.

To run the full 20-fold training for the **new taxonomy**:

```bash
bash scripts/best_hp_20folds.sh
```

To run for the **Sánchez taxonomy**:

```bash
bash scripts/best_hp_sanchez_tax_20folds.sh
```

---

## Inference

Inference is automatically triggered at the end of each training script — no separate step is required for standard evaluation.

---

## Project Structure

```
HBRF/
├── model.py                   # HBRF model definition
├── training.py                # Main training entry point
├── scripts/
│   ├── best_hp_20folds.sh               # New taxonomy training (20 folds)
│   ├── best_hp_sanchez_tax_20folds.sh   # Sánchez taxonomy training (20 folds)
│   └── search_hp.sh                     # Hyperparameter search script
├── utils/
│   ├── astro_objects.py       # AstroObject loading and handling utilities
│   └── features_processing.py # Feature extraction and preprocessing
└── results/
    ├── 250408_ndetge8/                      # 5-fold results
    ├── 250408_ndetge8_20folds/              # 20-fold results, new taxonomy
    └── 250408_ndetge8_sanchez_tax_20folds/  # 20-fold results, Sánchez taxonomy
```

## Results

Training outputs and model checkpoints are saved under `results/`. Evaluation notebooks with confusion matrices and performance metrics for each taxonomy are available in the parent directory `ztf_ff/notebooks/`.