from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


@dataclass
class AstroObject:
    metadata: pd.DataFrame
    detections: pd.DataFrame
    non_detections: [Optional[pd.DataFrame]] = None
    forced_photometry: [Optional[pd.DataFrame]] = None
    xmatch: [Optional[pd.DataFrame]] = None
    stamps: Optional[Dict[str, np.ndarray]] = None  # Might change
    features: [Optional[pd.DataFrame]] = None
    predictions: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if 'aid' not in self.metadata['name'].values:
            raise ValueError("'aid' is a mandatory field of metadata")

        mandatory_detection_columns = {
            'candid', 'tid', 'mjd', 'sid',
            'fid', 'pid', 'ra', 'dec', 'brightness',
            'e_brightness', 'unit'}

        missing_detections_columns = mandatory_detection_columns - set(self.detections.columns)
        if len(missing_detections_columns) > 0:
            raise ValueError(f"detections has missing columns: {missing_detections_columns}")

        if self.features is None:
            self.features = empty_normal_dataframe()

        if self.predictions is None:
            self.predictions = empty_normal_dataframe()


def empty_normal_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            'name',
            'value',
            'fid',
            'sid',
            'version'
        ]
    )
    return df
