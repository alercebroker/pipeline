"""Helpers shared by every survey: feature rows for the db and the output, WISE magnitudes, last mjd."""
import logging  # warning for features missing from the LUT

import numpy as np  # NaN and infinities
import pandas as pd  # feature frames

from features.surveys.base import SurveyBands  # band table type

DETECTION_KEYS_MAP = {  # input key -> AstroObject column, shared by ztf and lsst
    "oid": "oid",  # object id
    "sid": "sid",  # survey id
    "mjd": "mjd",  # epoch time
    "ra": "ra",  # right ascension
    "dec": "dec",  # declination
    "tid": "tid",  # telescope id
    "band": "fid",  # LSST usa 'band', ZTF usa 'fid'
    "measurement_id": "candid",  # LSST 'measurement_id' ≈ ZTF 'candid'
    "candid": "candid",  # ztf legacy spelling
    "aid": "aid",  # alerce id
    "fid": "fid",  # ZTF 'fid' ≈ LSST 'band'
    "isdiffpos": "isdiffpos",  # sign of the difference flux
    "forced": "forced",  # forced photometry flag
    "pid": "pid",  # processing id
}  # end of the keys map


def wise_magnitudes(xmatches: dict | None) -> tuple[float, float, float, float]:
    """Return W1-W4 from an allwise xmatch, or four NaN when there is none or it is another catalog."""
    if xmatches is not None and xmatches.get("catalog") == "allwise":  # only allwise carries w<n>mpro
        metadata = xmatches["metadata"]  # catalog columns of the match
        return tuple(metadata[f"w{n}mpro"]["Float64"] for n in (1, 2, 3, 4))  # W1, W2, W3, W4
    return (np.nan, np.nan, np.nan, np.nan)  # no allwise match


def last_mjd_of(detections: list[dict], forced: list[dict]) -> float:
    """Return the max mjd over detections and forced photometry, NaN when neither has one."""
    all_mjds = [det["mjd"] for det in detections if "mjd" in det]  # detection mjds
    all_mjds += [frc["mjd"] for frc in forced if "mjd" in frc]  # forced photometry mjds
    return float(max(all_mjds)) if all_mjds else np.nan  # NaN when empty


def prepare_features_for_db(  # signature spans three lines
    astro_object, feature_name_lut: dict, bands: SurveyBands, keep_name: bool  # object, id -> name, band table, name flag
) -> pd.DataFrame:  # returns the feature rows frame
    """Return the feature rows of `astro_object` for the scribe upsert.

    Keeps rows with a value, maps fid to the db band code, turns inf into None,
    applies the Power_rate backward-compat names, maps names to LUT ids and
    drops (with a warning) names the LUT does not know. Drops fid, and name
    too when `keep_name` is False.
    """
    ao_features = astro_object.features[["name", "fid", "value"]].copy()  # only the columns we need
    ao_features = ao_features[ao_features["value"].notna()]  # NaN values are not stored

    ao_features["band"] = ao_features["fid"].apply(bands.fid_for_db)  # band letter -> db code, unknown -> 0
    ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)  # inf -> None

    ao_features["name"] = ao_features["name"].replace(  # backward compatibility names
        {  # old name -> db name
            "Power_rate_1_4": "Power_rate_1/4",  # name stored in the db
            "Power_rate_1_3": "Power_rate_1/3",  # name stored in the db
            "Power_rate_1_2": "Power_rate_1/2",  # name stored in the db
        }  # end of the rename map
    )  # end of the replace call

    name_to_id = {name: feature_id for feature_id, name in feature_name_lut.items()}  # invert the LUT
    ao_features["feature_id"] = ao_features["name"].map(name_to_id)  # name -> feature id

    # feature_id is NOT NULL and part of the PK in `feature`: one unmapped row would fail the batch
    unmapped = ao_features["feature_id"].isna()  # names the LUT does not know
    if unmapped.any():  # at least one unknown name
        logging.getLogger("alerce.FeatureStep").warning(  # same logger as today
            f"Features not found in lookup table: {list(ao_features.loc[unmapped, 'name'].unique())}"  # list the names
        )  # end of the warning call
        ao_features = ao_features[~unmapped]  # drop them
    ao_features["feature_id"] = ao_features["feature_id"].astype(int)  # ids as ints

    drop = ["fid"] if keep_name else ["fid", "name"]  # ztf keeps name for the object update
    return ao_features.drop(columns=drop)  # mapped rows only


def flatten_features_for_output(astro_object, bands: SurveyBands) -> dict[str, float | None]:
    """Return the features of `astro_object` as {name + band suffix: value} for the output topic.

    Sorted by suffixed name, '-' replaced by '_', NaN and inf become None,
    other values are floats. An unknown or missing band adds no suffix.
    """
    ao_features = astro_object.features[["name", "fid", "value"]].copy()  # only the columns we need
    ao_features["name"] = ao_features["name"] + ao_features["fid"].map(bands.suffix)  # append "_<code>" or ""
    ao_features = ao_features.sort_values("name")  # sorted by suffixed name
    ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)  # NaN and inf -> None
    feature_names = [f.replace("-", "_") for f in ao_features["name"].values]  # g-r_max -> g_r_max
    values = ao_features["value"].astype(float)  # None -> NaN, everything else float
    return {  # NaN -> None, the rest as python float
        name: (None if np.isnan(value) else float(value))  # one entry per feature
        for name, value in zip(feature_names, values)  # pair names with values
    }  # end of the features dict
