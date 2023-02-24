from typing import Union
import pickle
import pandas as pd
import numpy as np


def _parse_rfid(rfid: Union[None, str]):
    return None if rfid is None else int(rfid)


def _parse_extra_fields(extra_fields: dict):
    extra_fields = extra_fields.copy()
    for k in extra_fields.keys():  # transform to bytes if datatype is list
        if isinstance(extra_fields[k], list):
            extra_fields[k] = pickle.dumps(extra_fields[k])
    return extra_fields


def _parse_stamps(stamps: dict, tid: str) -> dict:
    """
    Parses the stamps to the output schema.
    Returns always the 3 stamps, but for ATLAS the template is None

    Input:
        {
            'cutoutScience': b'science',
            'cutoutTemplate': b'template',
            'cutoutDifference': b'difference'
        }

    Output:
        {
            'science': b'science',
            'template': b'science', # or None
            'difference': b'difference'
        }
    """

    def parse_ztf(stamp: dict):
        parsed_stamp = {
            "science": stamp["cutoutScience"],
            "template": stamp["cutoutTemplate"],
            "difference": stamp["cutoutDifference"],
        }
        return parsed_stamp

    def parse_atlas(stamp: dict):
        parsed_stamp = {
            "science": stamp["cutoutScience"],
            "template": None,
            "difference": stamp["cutoutDifference"],
        }
        return parsed_stamp

    if tid.lower().startswith("ztf"):
        return parse_ztf(stamps)

    if tid.lower().startswith("atlas"):
        return parse_atlas(stamps)
    return stamps


def parse_output(alert: pd.Series) -> dict:
    """
    Transforms a pandas.Series representing a single alert, where each column
    is an attribute of the alert, to a dictionary with the proper values in
    each key.

    Input:
        oid             ZTF_ALERT1
        tid             ZTF
        pid             3047489
        candid          4307715
        mjd             58331.5
        fid             1
        ra              97.483553
        dec             13.70292
        rb              0.30679
        rbversion       v1
        mag             16.708749
        e_mag           0.64876
        rfid            7822447.0
        isdiffpos       -1
        e_ra            0.065
        e_dec           .065
        extra_fields    {'prv_candidates': None}
        stamps          {'cutoutScience': b'science', 'cutoutTemplate'...
        aid             AL23ldpubxoagfumu

    Output:

    """
    alert = alert.replace({np.nan: None})
    alert_dict = alert.to_dict()
    alert_dict["rfid"] = _parse_rfid(alert_dict["rfid"])
    alert_dict["extra_fields"] = _parse_extra_fields(alert_dict["extra_fields"])
    alert_dict["stamps"] = _parse_stamps(alert_dict["stamps"], alert_dict["tid"])
    return alert_dict
