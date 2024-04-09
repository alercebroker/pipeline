from typing import List
from abc import ABC, abstractmethod

from tqdm import tqdm

from lc_classifier.base import AstroObject
# import psutil


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

            # Getting usage of virtual_memory in GB ( 4th field)
            # print('RAM Used (MB):', psutil.virtual_memory()[3] / 1000000, extractor)


class LightcurvePreprocessor(ABC):
    @abstractmethod
    def preprocess_single_object(self, astro_object: AstroObject):
        pass

    def preprocess_batch(self, astro_objects: List[AstroObject], progress_bar=False):
        for astro_object in tqdm(astro_objects, disable=(not progress_bar)):
            self.preprocess_single_object(astro_object)
