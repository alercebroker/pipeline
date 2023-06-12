from features.core._base import BaseFeatureExtractor


class ExtractorNotFoundException(Exception):
    
    def __init__(self, name) -> None:
        message = f"Extractor {name} does not exist"
        super().__init__(message)


def selector(name: str) -> type[BaseFeatureExtractor]:
    if name.lower() == "ztf":
        from features.core.ztf import ZTFFeatureExtractor
        return ZTFFeatureExtractor
    if name.lower() == "elasticc":
        from features.core.elasticc import ELAsTiCCFeatureExtractor
        return ELAsTiCCFeatureExtractor
    raise ExtractorNotFoundException(name)
