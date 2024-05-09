import numpy as np
import pandas as pd
from . import FeatureFunctionLib
from numba import jit
import warnings


@jit(nopython=True)
def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


class FeatureSpace(object):
    def __init__(self, feature_list, extra_arguments: dict = {}):
        self.feature_objects = []
        self.feature_names = []
        self.shared_data = {}
        self.data_column_names = ["brightness", "mjd", "e_brightness"]

        for feature_name in feature_list:
            feature_class = getattr(FeatureFunctionLib, feature_name)
            if feature_name in extra_arguments:
                feature_instance = feature_class(
                    self.shared_data, **extra_arguments[feature_name]
                )
            else:
                feature_instance = feature_class(self.shared_data)
            self.feature_objects.append(feature_instance)
            self.feature_names.append(feature_name)

    def __lightcurve_to_array(self, lightcurve):
        return lightcurve[self.data_column_names].values.T

    def calculate_features(self, observations: pd.DataFrame):
        if len(observations) <= 5:
            features = []
            for name in self.feature_names:
                features.append((name, np.nan))

            return features

        lightcurve_array = self.__lightcurve_to_array(observations)

        features = []
        self.shared_data.clear()
        for name, feature_object in zip(self.feature_names, self.feature_objects):
            try:
                result = feature_object.fit(lightcurve_array)
            except Exception as e:
                warnings.warn(f"Exception when computing turbo-fats feature: {e}")
                result = np.NaN

            features.append((name, result))
        return features
