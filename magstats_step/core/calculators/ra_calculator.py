import pandas as pd
from ..factories.object import AlerceObject
from .coordinate_calculator import calculate_stats_coordinates


def calculate_ra(
    alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame
):
    ra_series, e_ra_series = detections["ra"], detections["e_ra"]
    meanra, ra_error = calculate_stats_coordinates(ra_series, e_ra_series)
    alerce_object.meanra = meanra
    alerce_object.sigmara = ra_error

    return alerce_object, detections, non_detections
