import numpy as np
import random
import cProfile

from Feature import *


"""
    This function initializes all the data needed to run the full suite of functions included in the library.
    Returns np.array
"""
def set_variables():
    time = np.random.choice(15000, 10000, replace=False)
    time.sort()
    mag = np.random.normal(loc=0.0, scale=1, size=len(time))
    error = np.random.normal(loc=0.01, scale=0.8, size=len(time))

    time2 = np.random.choice(15000, 10000, replace=False)
    time2.sort()
    mag2 = np.random.normal(loc=0.0, scale=1, size=len(time))
    error2 = np.random.normal(loc=0.01, scale=0.8, size=len(time))

    aligned_time = time
    aligned_time2 = time2
    aligned_mag = mag
    aligned_mag2 = mag2
    aligned_error = error
    aligned_error2 = error2

    lc = np.array([mag, time, error, mag2, aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2])
    return lc

data_all = set_variables()

def run_all_functions(data= data_all):
    a = FeatureSpace(Data='all', featureList=None, excludeList=["interp1d", "CAR_sigma"])
    a.calculateFeature(data)


cProfile.run("run_all_functions(data_all)", sort='cumtime')