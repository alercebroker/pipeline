import pandas as pd
from ..factories.object import AlerceObject


def calculate_dmdt(
    alerce_object: AlerceObject, detections: pd.DataFrame, non_detections: pd.DataFrame
):
    return alerce_object, detections, non_detections
