import pandas as pd
from ..factories.object import AlerceObject
from .coordinate_calculator import calculate_stats_coordinates


def calculate_dec(
    alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame
):
    dec_series, e_dec_series = detections["dec"], detections["e_dec"]
    meandec, dec_error = calculate_stats_coordinates(dec_series, e_dec_series)

    alerce_object.meandec = meandec
    alerce_object.sigmadec = dec_error

    return alerce_object, detections, non_detections
