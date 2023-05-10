import random
import numpy

import pandas as pd

random.seed(7193, version=2)


def generate_features_df(messages: pd.DataFrame):

    feature_columns = [
        ("Amplitude", 1),
        ("Amplitude", 2),
        ("Multiband_period", 12),
        ("feat3", -99),
        ("rb", 0),
    ]
    feature_multi_index = pd.MultiIndex.from_tuples(feature_columns)

    aids = messages["aid"].to_list()
    features_data = map(
        lambda x: numpy.random.uniform(1000.0, 9999.9, len(feature_columns)), aids
    )

    features_df = pd.DataFrame(features_data, index=aids, columns=feature_multi_index)
    features_df.index.name = "aid"

    return features_df
