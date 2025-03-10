from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ParserInterface(ABC):
    @abstractmethod
    def parse(self, messages: list[dict[str, Any]]):
        pass

    @abstractmethod
    def get_common_objects(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_survey_objects(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_common_detections(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_survey_detections(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_survey_non_detections(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_common_forced_photometries(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_survey_forced_photometries(self) -> pd.DataFrame:
        pass
