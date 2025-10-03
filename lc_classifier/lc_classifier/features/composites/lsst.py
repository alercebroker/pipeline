from typing import List


from ..extractors.dummy_extractor import DummyExtractor
from ..core.base import FeatureExtractorComposite, FeatureExtractor



class LSSTFeatureExtractor(FeatureExtractorComposite):
    version = "1.0.1"

    def _instantiate_extractors(self) -> List[FeatureExtractor]:
        bands = list("gr")

        feature_extractors = [
            DummyExtractor()
        ]
        return feature_extractors
