from dataclasses import dataclass
from typing import Any, List
from _types import *

@dataclass
class InputDTO:
    detections: Detections #includes forced photometry
    non_detections: NonDetections
    features: Features
    xmatch: Xmatch
    stamps: Stamps

    def __getattribute__(self, name: str):
        __dict__ = super(InputDTO, self).__getattribute__('__dict__')
        if name in __dict__:
            return super(InputDTO, self).__getattribute__(name)._value
        raise AttributeError

@dataclass
class OutputDTO:
    probabilities: pd.DataFrame