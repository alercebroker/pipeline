import numpy as np
import pandas as pd
from ..factories.object import AlerceObject


def _get_mag_from_index(df, idx):
    dets_from_index = df.loc[idx]
    mag = dets_from_index.mag
    mag.index = dets_from_index.fid
    return mag


def create_new_names(column_names, corr=False):
    corr_suffix = "_corr" if corr else ""
    return [f"mag{c if c != 'std' else 'sigma'}{corr_suffix}" for c in column_names]


def calculate_magnitude_statistics(alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame):
    detections_grouped = detections.groupby("fid")
    corrected_dets = detections[detections.corrected]
    corrected_dets_grouped = corrected_dets.groupby("fid")
    # Evil math trick because pandas std gives NaN with one element in a fid group
    def std(x):
        return np.std(x)

    stats = detections_grouped.agg(
        {"mag": ["min", "max", "median", "mean", std]}
    ).droplevel(level=0, axis=1)
    stats.columns = create_new_names(stats.columns)

    stats_corr = corrected_dets_grouped.agg(
        {"mag_corr": ["min", "max", "median", "mean", std]}
    ).droplevel(level=0, axis=1)
    stats_corr.columns = create_new_names(stats_corr.columns, corr=True)

    first = _get_mag_from_index(detections, detections_grouped.mjd.idxmin())
    last = _get_mag_from_index(detections, detections_grouped.mjd.idxmax())
    stats = stats.assign(magfirst=first, maglast=last)

    first_corr = _get_mag_from_index(corrected_dets, corrected_dets_grouped.mjd.idxmin())
    last_corr = _get_mag_from_index(corrected_dets, corrected_dets_grouped.mjd.idxmax())
    stats_corr = stats_corr.assign(magfirst_corr=first_corr, maglast_corr=last_corr)

    stats = stats.join(stats_corr)
    alerce_object.magstats = stats

    return alerce_object, detections, non_detections
