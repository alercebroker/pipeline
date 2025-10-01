import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import SkyCoord
from astropy import units as u

from ..core.base import LightcurvePreprocessor, AstroObject
from ..core.base import discard_bogus_detections


class ZTFLightcurvePreprocessor(LightcurvePreprocessor):
    def __init__(self, drop_bogus: bool = False):
        self.drop_bogus = drop_bogus

    def preprocess_single_object(self, astro_object: AstroObject):
       pass
