from features.core.elasticc import ELAsTiCCFeatureExtractor
from features.core.ztf import ZTFFeatureExtractor
from typing import Callable


class ExtractorNotFoundException(Exception):
    def __init__(self, name) -> None:
        message = f"Extractor {name} does not exist"
        super().__init__(message)


def selector(
    name: str,
) -> type[ZTFFeatureExtractor] | type[ELAsTiCCFeatureExtractor]:
    if name.lower() == "ztf":
        return ZTFFeatureExtractor
    if name.lower() == "elasticc":
        return ELAsTiCCFeatureExtractor
    raise ExtractorNotFoundException(name)
