import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import SkyCoord
from astropy import units as u

from lc_classifier.base import AstroObject
from ..core.base import LightcurvePreprocessor


class ZTFLightcurvePreprocessor(LightcurvePreprocessor):
    def preprocess_single_object(self, astro_object: AstroObject):
        self._helio_time_correction(astro_object)

    def _helio_time_correction(self, astro_object: AstroObject):
        detections = astro_object.detections
        ra_deg, dec_deg = detections[['ra', 'dec']].mean().values
        star_pos = SkyCoord(
            ra=ra_deg * u.degree,
            dec=dec_deg * u.degree,
            distance=1 * u.au,
            frame='icrs'
        )
        if np.isnan(ra_deg) or np.isnan(dec_deg):
            return

        def helio_correct_dataframe(dataframe: pd.DataFrame):
            times = Time(dataframe['mjd'], format='mjd')
            earth_pos = get_body_barycentric('earth', times)
            dots = earth_pos.dot(star_pos.cartesian)
            time_corrections = dots.value * (u.au.to(u.lightsecond)*u.second).to(u.day)
            dataframe['mjd'] += time_corrections.value

        helio_correct_dataframe(astro_object.detections)
        forced_photometry = astro_object.forced_photometry
        if forced_photometry is not None and len(forced_photometry) > 0:
            helio_correct_dataframe(forced_photometry)

        non_detections = astro_object.non_detections
        if non_detections is not None and len(non_detections) > 0:
            helio_correct_dataframe(non_detections)
