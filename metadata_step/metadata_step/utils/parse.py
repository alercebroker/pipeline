import numpy as np
from typing import Dict, List
from .constants import REFERENCE_KEYS, GAIA_KEYS, SS_KEYS, PS1_KEYS, DATAQUALITY_KEYS


def _none_to_nan(value):
    if value == None:
        return np.nan
    return value


def _filter_keys(d: Dict, keys: List):
    return {k: _none_to_nan(v) for k, v in d.items() if k in keys}


def _is_close(a, b):
    a = float(a)
    b = float(b)

    return np.isclose(a, b, atol=1e-03, rtol=1e-03)


# only needs to filter keys
def format_ss(alert: Dict):
    alert = _filter_keys(alert, SS_KEYS)
    return alert


# only needs to filter keys
def format_dataquality(alert: Dict):
    alert = _filter_keys(alert, DATAQUALITY_KEYS)
    return alert


# filter + calculate mjd
def format_reference(alert: Dict):
    new_alert = _filter_keys(alert, REFERENCE_KEYS)
    new_alert["mjdstartref"] = alert["jdstartref"] - 2400000.5
    new_alert["mjdendref"] = alert["jdendref"] - 2400000.5
    return new_alert


# needs past alerts/database catalog
def format_gaia(alert: Dict, catalog={}):
    alert = _filter_keys(alert, GAIA_KEYS)
    # TODO: update "unique1" value when needed
    alert["unique1"] = True
    gaia = catalog.get(alert["oid"])
    if gaia:
        gaia = gaia[0]
        if _is_close(alert["maggaia"], gaia["maggaia"]) and _is_close(
            alert["maggaiabright"],
            gaia["maggaiabright"],
        ):
            alert["unique1"] = False

    return alert


# this one too
def format_ps1(alert: Dict, catalog={}):
    alert = _filter_keys(alert, PS1_KEYS)
    alert["unique1"] = True
    alert["unique2"] = True
    alert["unique3"] = True

    ps1: List = catalog.get(alert["oid"])
    new_ps1 = []
    if ps1:
        candids = {}
        for i in [1, 2, 3]:
            ps1_filtered = list(
                filter(
                    lambda x: _is_close(x[f"objectidps{i}"], alert[f"objectidps{i}"]),
                    ps1,
                )
            )
            if len(ps1_filtered):
                alert[f"unique{i}"] = False
            else:
                continue

            for ps1_aux in ps1_filtered:
                ps1_aux[f"unique{i}"] = False
                ps1_aux["updated"] = True

    return alert


# formats each alert to send it to scribe psql
def format_detection(alert: Dict, catalogs: Dict):
    return {
        "oid": alert["oid"],
        "ss": format_ss(alert),
        "reference": format_reference(alert),
        "dataquality": format_dataquality(alert),
        "gaia": format_gaia(alert, catalogs["gaia"]),
        "ps1": format_ps1(alert, catalogs["ps1"]),
    }
