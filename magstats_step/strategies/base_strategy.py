import abc
from typing import List


class BaseStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_magstats(self, detections: List[dict], non_detections: List[dict]):
        raise NotImplementedError()
