from .dto import InputDTO
from _types import *

def input_dto_factory(detections, non_detections, forced_photometry, features, xmatch, stamps) -> InputDTO:
    return InputDTO(
        Detections(detections),
        NonDetections(non_detections),
        ForcedPhotometry(forced_photometry),
        Features(features),
        Xmatch(xmatch),
        Stamps(stamps)
    )