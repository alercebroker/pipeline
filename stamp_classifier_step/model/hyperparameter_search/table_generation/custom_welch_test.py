import os
import sys

"""
first run ecaluating_models_to_get_results_pickle.py
and then this 
"""

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

import matplotlib

# matplotlib.use('Agg')
import numpy as np
import pandas as pd
from parameters import general_keys
import matplotlib.pyplot as plt
from hyperparameter_search.table_generation.imposing_folder_and_trainings_consistency import (
    string_to_int_or_float,
)
import re
from scipy import stats


if __name__ == "__main__":
    list_a = [0.926, 0.944, 0.936, 0.942, 0.94]
    list_b = [0.938, 0.944, 0.946, 0.942, 0.936]
    welchs_t_test = stats.ttest_ind(list_a, list_b, equal_var=False)[1]
    print(welchs_t_test)

    list_a = [
        0.02744269371032715,
        0.023442506790161133,
        0.026683330535888672,
        0.024587631225585938,
        0.021908283233642578,
    ]
    list_b = [
        0.019646883010864258,
        0.019567012786865234,
        0.019634723663330078,
        0.019266843795776367,
        0.019517183303833008,
    ]
    welchs_t_test = stats.ttest_ind(list_a, list_b, equal_var=False)[1]
    print(welchs_t_test)
