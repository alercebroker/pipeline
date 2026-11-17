"""Registry mapping the SURVEY name to its Survey class."""
from features.surveys.lsst import LSSTSurvey  # lsst survey class
from features.surveys.ztf import ZTFSurvey  # ztf survey class

SURVEYS = {"ztf": ZTFSurvey, "lsst": LSSTSurvey}  # name -> class


def get_survey(name):
    """Return a new Survey instance for name; ValueError if the name is unknown."""
    if name not in SURVEYS:  # unknown survey name
        raise ValueError(f"unknown SURVEY {name!r}; known: {sorted(SURVEYS)}")  # list known names
    return SURVEYS[name]()  # new stateless instance
