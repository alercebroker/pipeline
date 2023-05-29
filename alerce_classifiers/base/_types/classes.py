from dataclasses import dataclass
import pandas as pd


# make value a list of typed dictionaries
@dataclass
class Detections:
    _value: pd.DataFrame


@dataclass
class NonDetections:
    _value: pd.DataFrame


@dataclass
class Xmatch:
    _value: pd.DataFrame


@dataclass
class Features:
    _value: pd.DataFrame


@dataclass
class Stamps:
    _value: pd.DataFrame
