# ALeRCE Classifier Models

This repository provides the necessary steps to run the **ALeRCE classifier models**.

## Clone the Repository

Begin by cloning the repository:

```
git clone https://github.com/alercebroker/pipeline.git
```

You first should install the classifier libraries.

```
python -m pip install -e ./lc_classifier
python -m pip install -e ./alerce_classifiers
```

## Data Acquisition 

Navigate to the data acquisition directory:

```
cd pipeline/training/data_acquisition/ztf_forced_photometry
```

Refer to the [`README.MD`](data_acquisition/README.MD) to obtain the data.

**Outputs:** 

1. Folders containing pickle files with dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py). These dictionaries include the following keys: `'metadata'`, `'detections'`, `'non_detections'`, `'forced_photometry'`, `'xmatch'`, `'stamps'`, `'features'`, and `'predictions'`.

2. Parquet files containing partitions.

## Models

The models are organized into two folders: **experimentation** and **staging_production**.

1. **Experimentation:** This folder contains the source code for the models, which can be modified for experimental purposes and testing new ideas. The models in this folder use `AstroObjects` as their input format. Models available:

    * ATAT
    * classifiers (which contains Machine Learning algorithms, as HBRF, LightGBM, XGBoost, MLP)

2. **Staging_production:** This folder imports the models located in [`staging repository`](https://github.com/alercebroker/alerce_classifiers) for training and/or inference to validate their functionality. The models use [`InputDTO`](https://github.com/alercebroker/pipeline/blob/main/schemas/feature_step/output.avsc) as the input format. `AstroObjects` data, provided in the `data_acquisition folder` is parsed into `InputDTO` to ensure compatibility with the staging environment. Models available:

    * mbappe.
    * squidward.

Refer to the appropriate `README.md` file to run the models after obtaining the data. **If you add a new model, please update this section.**

## Notes

For additional support, refer to the documentation or contact the ALeRCE team.





