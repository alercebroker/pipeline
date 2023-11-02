# Development of lightcurve classifier for ZTF

Last update: 20231102

## Data

ZTF forced photometry obtained from the ZTF service in 2021. 
It has to be adapted to run with ZTF alert schema v4 (with forced photometry).

## Model

Feature computation (libs/lc_classifier) + MLP