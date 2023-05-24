from features.core._base import BaseFeatureExtractor


class ExtractorNotFoundException(Exception):
    
    def __init__(self, extractor_name, extractor_list) -> None:
        message = f"Extractor: {extractor_name} not found in: \n\t {extractor_list}"
        super().__init__(message)


def extractor_selector(extractor_name: str) -> BaseFeatureExtractor:
    AVIALABLE_EXTRACTORS = ["ZTF", "ELASTICC"]
    # el tipo de retorno es costoso en este caso, quizas no es necesario
    # otro comentario, quizas seria bueno hacer un enumerate (y pasar un int)
    # para los extractores o un to upper. La idea es hacerlo consistente.
    # el problema con enumerate es que es poco legible desde config.
    if extractor_name == "ZTF":
        from features.core.ztf import ZTFClassifierFeatureExtractor
        return ZTFClassifierFeatureExtractor
    if extractor_name == "ELASTICC":
        from features.core.elasticc import ELAsTiCCClassifierFeatureExtractor
        return ELAsTiCCClassifierFeatureExtractor
    else:
        raise ExtractorNotFoundException(extractor_name, AVIALABLE_EXTRACTORS)
