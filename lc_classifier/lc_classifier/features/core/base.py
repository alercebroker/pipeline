from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod

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


class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features_single_object(self, astro_object: AstroObject):
        """This method is inplace"""
        pass

    def compute_features_batch(self, astro_objects: List[AstroObject], progress_bar=False):
        for astro_object in tqdm(astro_objects, disable=(not progress_bar)):
            self.compute_features_single_object(astro_object)


class FeatureExtractorComposite(FeatureExtractor, ABC):
    def __init__(self):
        self.extractors = self._instantiate_extractors()

    @abstractmethod
    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        pass

    def compute_features_single_object(self, astro_object: AstroObject):
        for extractor in self.extractors:
            extractor.compute_features_single_object(astro_object)


class LightcurvePreprocessor(ABC):
    @abstractmethod
    def preprocess_single_object(self, astro_object: AstroObject):
        pass

    def preprocess_batch(self, astro_objects: List[AstroObject], progress_bar=False):
        for astro_object in tqdm(astro_objects, disable=(not progress_bar)):
            self.preprocess_single_object(astro_object)


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


def query_ao_table(table: pd.DataFrame, name: str, check_unique: bool = True):
    ans_df = table[table['name'] == name]
    if check_unique and len(ans_df) > 1:
        raise Exception(f'Field {name} appears {len(ans_df)} times.')

    if check_unique:
        return ans_df['value'].values[0]
    else:
        return ans_df
