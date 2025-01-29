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
        self._helio_time_correction(astro_object)
        self.drop_absurd_detections(astro_object)
        # TODO: does error need a np.maximum(error, 0.01) ?
        # the factor depends on the units

        if self.drop_bogus:
            self.drop_bogus_detections(astro_object)

    def _helio_time_correction(self, astro_object: AstroObject):
        detections = astro_object.detections
        ra_deg, dec_deg = detections[["ra", "dec"]].mean().values
        star_pos = SkyCoord(
            ra=ra_deg * u.degree,
            dec=dec_deg * u.degree,
            distance=1 * u.au,
            frame="icrs",
        )
        if np.isnan(ra_deg) or np.isnan(dec_deg):
            return

        def helio_correct_dataframe(dataframe: pd.DataFrame):
            times = Time(dataframe["mjd"], format="mjd")
            earth_pos = get_body_barycentric("earth", times)
            dots = earth_pos.dot(star_pos.cartesian)
            time_corrections = dots.value * (u.au.to(u.lightsecond) * u.second).to(
                u.day
            )
            dataframe["mjd_nohelio"] = dataframe["mjd"]
            dataframe["mjd"] += time_corrections.value

        helio_correct_dataframe(astro_object.detections)
        forced_photometry = astro_object.forced_photometry
        if forced_photometry is not None and len(forced_photometry) > 0:
            helio_correct_dataframe(forced_photometry)

        non_detections = astro_object.non_detections
        if non_detections is not None and len(non_detections) > 0:
            helio_correct_dataframe(non_detections)

    def drop_absurd_detections(self, astro_object: AstroObject):
        def drop_absurd(table):
            magnitude_mask = table["unit"] == "magnitude"
            mag_det = table[magnitude_mask]
            table = pd.concat(
                [
                    mag_det[
                        (
                            (mag_det["brightness"] < 30.0)
                            & (mag_det["brightness"] > 6.0)
                            & (mag_det["e_brightness"] < 1.0)
                        )
                    ],
                    table[~magnitude_mask],
                ],
                axis=0,
            )
            return table

        astro_object.detections = drop_absurd(astro_object.detections)
        astro_object.forced_photometry = drop_absurd(astro_object.forced_photometry)

    def drop_bogus_detections(self, astro_object: AstroObject):
        def drop_bogus_dets(table):
            keys = table.keys()
            table = table.to_dict("records")
            table = discard_bogus_detections(table)
            table = pd.DataFrame.from_records(table)
            if len(table) == 0:
                table = pd.DataFrame(columns=keys)
            return table

        astro_object.detections = drop_bogus_dets(astro_object.detections)
        if len(astro_object.detections) == 0:
            astro_object.forced_photometry = pd.DataFrame(
                columns=astro_object.forced_photometry.keys()
            )
        else:
            astro_object.forced_photometry = drop_bogus_dets(
                astro_object.forced_photometry
            )


class ShortenPreprocessor(LightcurvePreprocessor):
    def __init__(self, n_days: float):
        self.n_days = n_days

    def preprocess_single_object(self, astro_object: AstroObject):
        first_mjd = np.min(astro_object.detections["mjd"])
        max_mjd = first_mjd + self.n_days
        astro_object.detections = astro_object.detections[
            astro_object.detections["mjd"] <= max_mjd
        ]
        last_detection_mjd = np.max(astro_object.detections["mjd"])
        astro_object.forced_photometry = astro_object.forced_photometry[
            astro_object.forced_photometry["mjd"] < last_detection_mjd
        ]


class ShortenNDetPreprocessor(LightcurvePreprocessor):
    def __init__(self, n_dets: float):
        self.n_dets = n_dets

    def preprocess_single_object(self, astro_object: AstroObject):
        dets = astro_object.detections[
            astro_object.detections["unit"] == "diff_flux"
        ].copy()
        dets = dets.sort_values(by="mjd").iloc[0 : self.n_dets].copy()
        max_mjd = np.max(dets["mjd"])
        del dets

        astro_object.detections = astro_object.detections[
            astro_object.detections["mjd"] <= max_mjd
        ]
        astro_object.forced_photometry = astro_object.forced_photometry[
            astro_object.forced_photometry["mjd"] < max_mjd
        ]
