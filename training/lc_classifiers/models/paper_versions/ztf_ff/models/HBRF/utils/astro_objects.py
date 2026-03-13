import pickle
from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from tqdm import tqdm

def empty_normal_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["name", "value", "fid", "sid", "version"])

@dataclass
class AstroObject:
    metadata: pd.DataFrame
    detections: pd.DataFrame
    non_detections: Optional[pd.DataFrame] = None
    forced_photometry: Optional[pd.DataFrame] = None
    xmatch: Optional[pd.DataFrame] = None
    reference: Optional[pd.DataFrame] = None
    stamps: Optional[Dict[str, np.ndarray]] = None
    features: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if "aid" not in self.metadata["name"].values:
            raise ValueError("'aid' is a mandatory field of metadata")

        mandatory_detection_columns = {
            "candid", "tid", "mjd", "sid", "fid", "pid",
            "ra", "dec", "brightness", "e_brightness", "unit"
        }

        missing = mandatory_detection_columns - set(self.detections.columns)
        if missing:
            raise ValueError(f"detections has missing columns: {missing}")

        if self.features is None:
            self.features = empty_normal_dataframe()

        if self.predictions is None:
            self.predictions = empty_normal_dataframe()

    def to_dict(self) -> Dict:
        return {
            "metadata": self.metadata,
            "detections": self.detections,
            "non_detections": self.non_detections,
            "forced_photometry": self.forced_photometry,
            "xmatch": self.xmatch,
            "reference": self.reference,
            "stamps": self.stamps,
            "features": self.features,
            "predictions": self.predictions,
        }

def astro_object_from_dict(d: Dict) -> AstroObject:
    return AstroObject(
        metadata=d["metadata"],
        detections=d["detections"],
        non_detections=d.get("non_detections"),
        forced_photometry=d.get("forced_photometry"),
        xmatch=d.get("xmatch"),
        reference=d.get("reference"),
        stamps=d.get("stamps"),
        features=d.get("features"),
        predictions=d.get("predictions"),
    )