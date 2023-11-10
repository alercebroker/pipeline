# Development of lightcurve classifier for ZTF

Last update: 20231102

## Data

ZTF forced photometry obtained from the ZTF service in 2021. 
It has to be adapted to run with ZTF alert schema v4 (with forced photometry).

## Model

Feature computation (libs/lc_classifier) + MLP

## Steps:
 * Run download_training_set_ff.ipynb to download the data
 * Run dataset.py to transform lightcurves.pkl and objects.pkl to AstroObject.
 * Run compute_features.py
 * 