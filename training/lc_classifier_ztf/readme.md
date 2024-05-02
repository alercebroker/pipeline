# Development of lightcurve classifier for ZTF

Last update: 20240502

## Data

ZTF forced photometry obtained from the ZTF service in 2021. 
It has to be adapted to run with ZTF alert schema v4 (with forced photometry).

## Model

Feature computation (libs/lc_classifier) + classifier

# Hierarchical Random Forest, codename "Squidward"

Very similar model to the light curve classifier from Sánchez-Saez et al. 2021. 

## Steps:
 * Run download_training_set_ff.ipynb to download the data
 * Run dataset.py to transform lightcurves.pkl and objects.pkl to AstroObject.
 * Run compute_features.py
 * Run consolidate_features.py and partition_labels.py
 * Run training.py
 * Run evaluate_model.py
