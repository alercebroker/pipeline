import numpy as np
import pandas as pd

from ..core.base import LightcurvePreprocessor, AstroObject
from ..core.base import discard_bogus_detections
from .barycentric import barycentric_time_correction

# Standard ZTF exposure, used for rows without `exptime`.
ZTF_DEFAULT_EXPTIME_S = 30.0


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
        """Barycentric (not heliocentric, despite the name) correction: `mjd`
        becomes the mid-exposure BJD_TDB, the original stays in `mjd_nohelio`.

        ZTF `mjd` is the start of the exposure, in UTC.
        """
        barycentric_time_correction(
            astro_object, scale="utc", default_exptime_s=ZTF_DEFAULT_EXPTIME_S
        )

    def drop_absurd_detections(self, astro_object: AstroObject):
        def drop_absurd(table):
            magnitude_mask = table["unit"] == "magnitude"
            mag_det = table[magnitude_mask]
            table = pd.concat(
                [
                    mag_det[
                        (
                            (mag_det["brightness"] < 30.0) #queda igual
                            & (mag_det["brightness"] > 6.0) #queda igual
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

    def drop_bogus_detections(self, astro_object: AstroObject): #esto es placeholder para LSST
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
