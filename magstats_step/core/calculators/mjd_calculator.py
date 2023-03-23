import pandas as pd
from ..factories.object import AlerceObject


def calculate_mjd(
    alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame
):
    if not detections.empty:
        alerce_object.firstmjd = detections["mjd"].min()
        alerce_object.lastmjd = detections["mjd"].max()

    return alerce_object, detections, non_detections
