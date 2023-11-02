import numpy as np

from typing import Dict, List
from .constants import REFERENCE_KEYS, GAIA_KEYS, SS_KEYS, PS1_KEYS, DATAQUALITY_KEYS


def _clear_nan(value):
    if value == np.nan:
        return None
    return value


def _filter_keys(d: Dict, keys: List):
    return {k: _clear_nan(v) for k, v in d.items() if d in keys}


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
    return alert


# needs past alerts/database catalog
def format_gaia(alert: Dict, catalog = []):
    alert = _filter_keys(alert, GAIA_KEYS)
    # TODO: update "unique1" value when needed
    alert["unique1"] = True
    return alert


# this one too
def format_ps1(alert: Dict, catalog = []):
    alert = _filter_keys(alert, PS1_KEYS)
    # TODO: update "uniqueX" values when needed
    alert["unique1"] = True
    alert["unique2"] = True
    alert["unique3"] = True
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
