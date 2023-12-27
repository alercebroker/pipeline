import pickle
import pandas as pd
import numpy as np


def _parse_extra_fields(extra_fields: dict):
    extra_fields = extra_fields.copy()
    for k in extra_fields.keys():  # transform to bytes if datatype is list
        if isinstance(extra_fields[k], list):
            extra_fields[k] = pickle.dumps(extra_fields[k])
    return extra_fields


def parse_output(alert: pd.Series) -> dict:
    """
    Transforms a pandas.Series representing a single alert, where each column
    is an attribute of the alert, to a dictionary with the proper values in
    each key.
    """
    alert = alert.replace({np.nan: None})
    alert_dict = alert.to_dict()
    alert_dict["extra_fields"] = _parse_extra_fields(
        alert_dict["extra_fields"]
    )
    return alert_dict
