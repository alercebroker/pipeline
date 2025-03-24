from abc import ABC, abstractmethod
from typing import Any, TypedDict

import pandas as pd


class ParsedData(TypedDict):
    """
    Dictionary of pandas `DataFrame`.

    The dictionary contains data parsed from the kafka messages associated
    with: objects, detections, non_detections and forced_photometries. For each
    there is a `DataFrame` that contains all columns needed for DB insertion.

    In cases where there are common and specific versions of the table, both
    sets of columns are stored on the dataframe.

    If the classification doesn't apply to a survey, for example, a survey
    dosen't have `non_detections`, then the associated `DataFrame` will be empty.
    """

    objects: pd.DataFrame
    detections: pd.DataFrame
    non_detections: pd.DataFrame
    forced_photometries: pd.DataFrame


class ParserInterface(ABC):
    """
    Minimal interface for a common survey parser.

    An implementation of this Interface is returned by the `Selector` to be
    used by the `Step` to parse the incoming messages.
    """

    @abstractmethod
    def parse(self, messages: list[dict[str, Any]]) -> ParsedData:
        """
        Parses a list of messages into a `ParsedData` dict of pandas DataFrames.
        """
        pass
