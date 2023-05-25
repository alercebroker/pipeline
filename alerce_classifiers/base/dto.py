from dataclasses import dataclass
from ._types import *


@dataclass
class InputDTO:
    detections: Detections  # includes forced photometry
    non_detections: NonDetections
    features: Features
    xmatch: Xmatch
    stamps: Stamps

    def __getattribute__(self, name: str):
        __dict__ = super(InputDTO, self).__getattribute__("__dict__")
        __class__ = super(InputDTO, self).__getattribute__("__class__")
        if name == "__class__":
            return __class__
        if name in __dict__:
            return super(InputDTO, self).__getattribute__(name)._value
        raise AttributeError


@dataclass
class OutputDTO:
    class_name: str
    probability: str
