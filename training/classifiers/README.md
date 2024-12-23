# ALeRCE Classifier Models

This repository provides the necessary steps to run the **ALeRCE classifier models**.

## Clone the Repository

Begin by cloning the repository:

```
git clone https://github.com/alercebroker/pipeline.git
```

You should install the dependencies (WORK IN PROGRESS).

## Data Acquisition 

Navigate to the data acquisition directory:

```
cd pipeline/training/data_acquisition/ztf_forced_photometry
```

Refer to the [`README.MD`](https://github.com/alercebroker/pipeline/tree/main/training/data_acquisition/ztf_forced_photometry) to obtain the data.

**Outputs:** 

1. Folders containing pickle files with dictionaries derived from instances of [`AstroObjects`](https://github.com/alercebroker/pipeline/blob/main/lc_classifier/lc_classifier/features/core/base.py). These dictionaries include the following keys: `'metadata'`, `'detections'`, `'non_detections'`, `'forced_photometry'`, `'xmatch'`, `'stamps'`, `'features'`, and `'predictions'`.

2. Parquet files containing partitions.

## Models

The models are organized into two folders: **experimentation** and **stagging_production**.

1. **Experimentation:** This folder contains the source code for the models, which can be modified for experimental purposes and testing new ideas. The models in this folder use `AstroObjects` as their input format. Models available:

    * ATAT (Mbappe)
    * HBRF (Squidward)
    * MLP (WORK IN PROGRESS).
    * LightGBM (WORK IN PROGRESS).

2. **Stagging_production:** This folder imports the models located in [`stagging repository`](https://github.com/alercebroker/alerce_classifiers) for training and/or inference to validate their functionality. The models use [`InputDTO`](https://github.com/alercebroker/pipeline/blob/main/schemas/feature_step/output.avsc) as the input format. `AstroObjects` data, provided in the [`data_acquisition folder`](https://github.com/alercebroker/pipeline/tree/main/training/classifiers/data_acquisition) is parsed into `InputDTO` to ensure compatibility with the stagging environment. Models available:

    * ATAT (Mbappe) (WORK IN PROGRESS).
    * HBRF (Squidward) (WORK IN PROGRESS).

Refer to the appropriate `README.md` file to run the models after obtaining the data. **If you add a new model, please update this section.**

## Notes

For additional support, refer to the documentation or contact the ALeRCE team.





