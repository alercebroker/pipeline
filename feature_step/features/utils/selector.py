from features.core.elasticc import ELAsTiCCFeatureExtractor
from features.core.ztf import ZTFFeatureExtractor
from typing import Callable


class ExtractorNotFoundException(Exception):
    def __init__(self, name) -> None:
        message = f"Extractor {name} does not exist"
        super().__init__(message)


def extractor_factory(
    name: str,
) -> type[ZTFFeatureExtractor] | Callable[..., ELAsTiCCFeatureExtractor]:
    if name.lower() == "ztf":
        return ZTFFeatureExtractor
    if name.lower() == "elasticc":
        from lc_classifier.features.preprocess.preprocess_elasticc import (
            ElasticcPreprocessor,
        )
        from lc_classifier.features.custom.elasticc_feature_extractor import (
            ElasticcFeatureExtractor,
        )

        def factory(
            detections, non_detections, xmatch
        ) -> ELAsTiCCFeatureExtractor:
            return ELAsTiCCFeatureExtractor(
                ElasticcPreprocessor(),
                ElasticcFeatureExtractor(round=2),
                detections,
                non_detections=non_detections,
                xmatch=xmatch,
            )

        return factory
    raise ExtractorNotFoundException(name)
