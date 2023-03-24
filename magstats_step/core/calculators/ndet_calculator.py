import pandas as pd
from ..factories.object import AlerceObject


def calculate_ndet(
    alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame
):
    alerce_object.ndet = len(detections.index)
    return alerce_object, detections, non_detections
