import numpy as np
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

from dataclasses import dataclass
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

    def __post_init__(self):
        if 'aid' not in self.metadata['field'].values:
            raise ValueError("'aid' is a mandatory field of metadata")
        if self.features is None:
            self.features = pd.DataFrame(
                columns=[
                    'name',
                    'value',
                    'fid',
                    'sid',
                    'version'
                ]
            )


class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features_single_object(self, astro_object: AstroObject):
        """This method is inplace"""
        pass

    def compute_features_batch(self, astro_objects: List[AstroObject]):
        for astro_object in astro_objects:
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
