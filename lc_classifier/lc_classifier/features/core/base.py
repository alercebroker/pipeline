import pickle
from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import time
import json
import hashlib
import os
from datetime import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm


@dataclass
class AstroObject:
    metadata: pd.DataFrame
    detections: pd.DataFrame
    non_detections: [Optional[pd.DataFrame]] = None
    forced_photometry: [Optional[pd.DataFrame]] = None
    xmatch: [Optional[pd.DataFrame]] = None
    reference: [Optional[pd.DataFrame]] = None
    stamps: Optional[Dict[str, np.ndarray]] = None  # Might change
    features: [Optional[pd.DataFrame]] = None
    predictions: Optional[pd.DataFrame] = None

    def __post_init__(self):

        mandatory_detection_columns = { #condicional al survey
            "oid",
            "sid",
            'fid',
        }

        missing_detections_columns = mandatory_detection_columns - set(
            self.detections.columns
        )
        if len(missing_detections_columns) > 0:
            raise ValueError(
                f"detections has missing columns: {missing_detections_columns}"
            )

        if self.features is None:
            self.features = empty_normal_dataframe()

        if self.predictions is None:
            self.predictions = empty_normal_dataframe()

    def to_dict(self) -> Dict:
        d = {
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
        return d


class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features_single_object(self, astro_object: AstroObject):
        """This method is inplace"""
        pass

    def compute_features_batch(
        self, astro_objects: List[AstroObject], progress_bar=False
    ):
        for astro_object in tqdm(astro_objects, disable=(not progress_bar)):
            self.compute_features_single_object(astro_object)


class FeatureExtractorComposite(FeatureExtractor, ABC):
    def __init__(self):
        self.extractors = self._instantiate_extractors()

    @abstractmethod
    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        pass

    def compute_features_single_object(self, astro_object: AstroObject):
        # Create directory if it doesn't exist
        output_dir = "extractor_timings"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get OID from metadata or detections
        if hasattr(astro_object.metadata, 'oid'):
            oid = astro_object.metadata.oid
        elif 'oid' in astro_object.metadata.columns:
            oid = astro_object.metadata['oid'].iloc[0] if len(astro_object.metadata) > 0 else "unknown"
        elif 'oid' in astro_object.detections.columns:
            oid = astro_object.detections['oid'].iloc[0] if len(astro_object.detections) > 0 else "unknown"
        else:
            oid = "unknown"
        
        # Get class name from astro_object
        class_name = getattr(astro_object, 'class_name', 'unknown')
        len_detections = getattr(astro_object, 'len_detections', 0)
        
        # Prepare timing data structure
        timing_data = {
            'oid': str(oid),
            'class_name': str(class_name),
            'timestamp': datetime.now().isoformat(),
            'detections': len_detections,
            'extractors': []
        }
        
        # Execute each extractor and time it
        for extractor in self.extractors:
            extractor_name = extractor.__class__.__name__
            start_time = time.time()
            
            extractor.compute_features_single_object(astro_object)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            timing_data['extractors'].append({
                'name': extractor_name,
                'elapsed_seconds': elapsed_time
            })
        
        # Generate filename with timestamp and hash
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        hash_str = hashlib.md5(f"{oid}_{timestamp_str}".encode()).hexdigest()[:8]
        filename = f"{timestamp_str}_{hash_str}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save timing data to JSON
        with open(filepath, 'w') as f:
            json.dump(timing_data, f, indent=2)

class LightcurvePreprocessor(ABC):
    @abstractmethod
    def preprocess_single_object(self, astro_object: AstroObject):
        pass

    def preprocess_batch(self, astro_objects: List[AstroObject], progress_bar=False):
        for astro_object in tqdm(astro_objects, disable=(not progress_bar)):
            self.preprocess_single_object(astro_object)


def discard_bogus_detections(detections: List[Dict]) -> list[dict]:
    RB_THRESHOLD = 0.55

    filtered_detections = []

    for det in detections:
        bogus = False

        if "extra_fields" in det.keys():
            rb = (
                det["extra_fields"]["rb"]
                if "rb" in det["extra_fields"].keys()
                else None
            )
            procstatus = (
                det["extra_fields"]["procstatus"]
                if "procstatus" in det["extra_fields"].keys()
                else None
            )
        else:
            rb = det["rb"] if "rb" in det.keys() else None
            procstatus = det["procstatus"] if "procstatus" in det.keys() else None

        mask_rb = rb is not None and not det["forced"] and (rb < RB_THRESHOLD)
        mask_procstatus = (
            procstatus is not None
            and det["forced"]
            and (procstatus != "0")
            and (procstatus != "57")
        )
        if mask_rb or mask_procstatus:
            bogus = True

        if not bogus:
            filtered_detections.append(det)

    return filtered_detections


def empty_normal_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(columns=["name", "value", "fid", "sid", "version"])
    return df


def astro_object_from_dict(d: Dict) -> AstroObject:
    astro_object = AstroObject(
        metadata=d["metadata"],
        detections=d["detections"],
        non_detections=d["non_detections"],
        forced_photometry=d["forced_photometry"],
        xmatch=d["xmatch"],
        reference=d["reference"],
        stamps=d["stamps"],
        features=d["features"],
        predictions=d["predictions"],
    )
    return astro_object


def query_ao_table(table: pd.DataFrame, name: str, check_unique: bool = True):
    ans_df = table[table["name"] == name]
    if check_unique and len(ans_df) > 1:
        raise Exception(f"Field {name} appears {len(ans_df)} times.")

    if check_unique:
        return ans_df["value"].values[0]
    else:
        return ans_df


def save_astro_objects_batch(astro_objects: List[AstroObject], filename: str):
    astro_objects_dicts = [ao.to_dict() for ao in astro_objects]
    with open(filename, "wb") as f:
        pickle.dump(astro_objects_dicts, f)
