import numpy as np

MAGSTATS_TRANSLATE = {
    "magpsf_mean": "magmean",
    "magpsf_median": "magmedian",
    "magpsf_max": "magmax",
    "magpsf_min": "magmin",
    "sigmapsf": "magsigma",
    "magpsf_last": "maglast",
    "magpsf_first": "magfirst",
    "magpsf_corr_mean": "magmean_corr",
    "magpsf_corr_median": "magmedian_corr",
    "magpsf_corr_max": "magmax_corr",
    "magpsf_corr_min": "magmin_corr",
    "sigmapsf_corr": "magsigma_corr",
    "magpsf_corr_last": "maglast_corr",
    "magpsf_corr_first": "magfirst_corr",
    "first_mjd": "firstmjd",
    "last_mjd": "lastmjd",
}

MAGSTATS_UPDATE_KEYS = [
    "stellar",
    "corrected",
    "ndet",
    "ndubious",
    "dmdt_first",
    "dm_first",
    "sigmadm_first",
    "dt_first",
    "magmean",
    "magmedian",
    "magmax",
    "magmin",
    "magsigma",
    "maglast",
    "magfirst",
    "magmean_corr",
    "magmedian_corr",
    "magmax_corr",
    "magmin_corr",
    "magsigma_corr",
    "maglast_corr",
    "magfirst_corr",
    "firstmjd",
    "lastmjd",
    "step_id_corr",
]

DISTANCE_THRESHOLD = 1.4  #: max threshold for distnr
SCORE_THRESHOLD = 0.4  #: max threshold for sgscore
CHINR_THRESHOLD = 2  #: max threshold for chinr
SHARPNR_MAX = 0.1  #: max value for sharpnr
SHARPNR_MIN = -0.13  #: min value for sharpnr
ZERO_MAG = 100.  #: default value for zero magnitude (a big value!)
TRIPLE_NAN = (np.nan, np.nan, np.nan)
MAGNITUDE_THRESHOLD = 13.2
