from features.core.elasticc import ELAsTiCCFeatureExtractor
from features.core.atlas import AtlasFeatureExtractor
from features.core.ztf import ZTFFeatureExtractor
from typing import Callable


class ExtractorNotFoundException(Exception):
    def __init__(self, name) -> None:
        message = f"Extractor {name} does not exist"
        super().__init__(message)


def selector(
    name: str,
) -> (
    type[ZTFFeatureExtractor]
    | type[ELAsTiCCFeatureExtractor]
    | type[AtlasFeatureExtractor]
):
    if name.lower() == "ztf":
        return ZTFFeatureExtractor
    if name.lower() == "elasticc":
        return ELAsTiCCFeatureExtractor
    if name.lower() == "atlas":
        return AtlasFeatureExtractor
    raise ExtractorNotFoundException(name)
