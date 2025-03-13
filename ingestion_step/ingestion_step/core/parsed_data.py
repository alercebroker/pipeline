from typing import TypedDict

import pandas as pd


class ParsedData(TypedDict):
    objects: pd.DataFrame
    detections: pd.DataFrame
    non_detections: pd.DataFrame
    forced_photometries: pd.DataFrame
