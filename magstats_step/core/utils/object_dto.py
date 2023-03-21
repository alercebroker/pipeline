from dataclasses import dataclass


@dataclass
class ObjectDTO:
    """Class which contains all the info needed to calculate all Alerce Object fields.
    All calculators must recieve an instance of this class
    """

    alerce_object: dict
    detections: list
    non_detections: list
