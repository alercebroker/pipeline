import pandas as pd
from dataclasses import dataclass


@dataclass
class ObjectDTO:
    """Class which contains all the info needed to calculate all Alerce Object fields.
    All calculators must recieve an instance of this class
    """

    alerce_object: dict
    detections: pd.DataFrame
    non_detections: pd.DataFrame
    extra_fields: pd.DataFrame
