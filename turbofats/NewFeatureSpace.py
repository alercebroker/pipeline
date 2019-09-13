import numpy as np
import pandas as pd
from turbofats import FeatureFunctionLib
from numba import jit


@jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i] :
            return False
    return True


class NewFeatureSpace(object):
    def __init__(self, feature_list, data_column_names=None):
        self.feature_objects = []
        self.feature_names = []
        if data_column_names is None:
            self.data_column_names = ['magpsf_corr', 'mjd', 'sigmapsf_corr']
        else:
            self.data_column_names = data_column_names

        for feature_name in feature_list:
            feature_class = getattr(FeatureFunctionLib, feature_name)
            feature_instance = feature_class()
            self.feature_objects.append(feature_instance)
            if feature_instance.is1d():
                self.feature_names.append(feature_name)
            else:
                self.feature_names += feature_instance.get_feature_names()

    def __lightcurve_to_array(self, lightcurve):
        return lightcurve[self.data_column_names].values.T

    def calculate_features(self, lightcurve):
        lightcurve = lightcurve.copy()
        if not is_sorted(lightcurve['mjd'].values):
            lightcurve.sort_values('mjd', inplace=True)
            
        lightcurve_array = self.__lightcurve_to_array(lightcurve)
                    
        self.results = []
        for feature in self.feature_objects:
            result = feature.fit(lightcurve_array)
            if feature.is1d():
                self.results.append(result)
            else:
                self.results += result
        self.results = np.array(self.results).reshape(1, -1).astype(np.float)
        df = pd.DataFrame(self.results, columns=self.feature_names)
        return df
