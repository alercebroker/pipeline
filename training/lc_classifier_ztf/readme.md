# Lightcurve classifier for ZTF forced photometry

Last update: 20240507

## Data

ZTF forced photometry obtained from the ZTF service in 2021. 
It has to be adapted to run with ZTF alert schema v4 (with forced photometry).

Examples per class in objects_with_wise_20240105.parquet

alerceclass
* AGN               2998
* Blazar            1478
* CEP               2998
* CV/Nova           1784
* DSCT              1659
* EA                3000
* EB/EW             3000
* LPV               3003
* Microlensing        55
* Periodic-Other    1334
* QSO               2999
* RRLab             2999
* RRLc              3000
* RSCVn             2578
* SLSN               207
* SNII              1611
* SNIIb              130
* SNIIn              301
* SNIa              2996
* SNIbc              549
* TDE                 87
* YSO               2998
* ZZ                  13


Examples per class in partitions.parquet

alerceclass
* AGN               2994
* Blazar            1477
* CEP               2995
* CV/Nova           1784
* DSCT              1658
* EA                3000
* EB/EW             3000
* LPV               3003
* Microlensing        36
* Periodic-Other    1334
* QSO               2998
* RRLab             2999
* RRLc              3000
* RSCVn             2578
* SLSN               207
* SNII              1611
* SNIIb              130
* SNIIn              301
* SNIa              2991
* SNIbc              549
* TDE                 87
* YSO               2996


## Model

Feature computation + classifier

# Hierarchical Random Forest, codename "Squidward"

Very similar model to the light curve classifier from Sánchez-Saez et al. 2021. 

# Results

See ./feature_computation/models/hrf_classifier_20240506-155647

## Steps:
 * Run download_training_set_ff.ipynb to download the data
 * Run dataset.py to transform lightcurves.pkl and objects.pkl to AstroObject.
 * Run compute_features.py
 * Run consolidate_features.py and partition_labels.py
 * Run training.py
 * Run evaluate_model.py
