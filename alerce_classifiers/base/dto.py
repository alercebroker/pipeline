from ._types import *
from dataclasses import dataclass


@dataclass
class InputDTO:
    _detections: Detections  # includes forced photometry
    _non_detections: NonDetections
    _features: Features
    _xmatch: Xmatch
    _stamps: Stamps

    @property
    def detections(self):
        return self._detections._value

    @property
    def non_detections(self):
        return self._non_detections._value

    @property
    def features(self):
        return self._features._value

    @property
    def xmatch(self):
        return self._xmatch._value

    @property
    def stamps(self):
        return self._stamps._value


@dataclass
class OutputDTO:
    """
    Parameters
    ----------
    probabilities: DataFrame
    aid     SN      AGN     Other
    aid1    0.5     0.4     0.1
    """

    probabilities: pd.DataFrame
