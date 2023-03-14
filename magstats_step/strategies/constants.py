import numpy as np

DISTANCE_THRESHOLD = 1.4  #: max threshold for distnr
SCORE_THRESHOLD = 0.4  #: max threshold for sgscore
CHINR_THRESHOLD = 2  #: max threshold for chinr
SHARPNR_MAX = 0.1  #: max value for sharpnr
SHARPNR_MIN = -0.13  #: min value for sharpnr
ZERO_MAG = 100.  #: default value for zero magnitude (a big value!)
TRIPLE_NAN = (np.nan, np.nan, np.nan)
MAGNITUDE_THRESHOLD = 13.2
