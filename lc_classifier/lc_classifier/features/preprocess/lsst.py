import numpy as np
import pandas as pd

from ..core.base import LightcurvePreprocessor, AstroObject
from ..core.base import discard_bogus_detections
from .barycentric import barycentric_time_correction


class LSSTLightcurvePreprocessor(LightcurvePreprocessor):
    def __init__(self, drop_bogus: bool = False):
        self.drop_bogus = drop_bogus

    def preprocess_single_object(self, astro_object: AstroObject):
        #tengo que poder correr esto
        self._helio_time_correction(astro_object)
        self.drop_absurd_detections(astro_object)
        # TODO: does error need a np.maximum(error, 0.01) ?
        # the factor depends on the units

        if self.drop_bogus:
            self.drop_bogus_detections(astro_object)

    def _helio_time_correction(self, astro_object: AstroObject):
        """Barycentric (not heliocentric, despite the name) correction: `mjd`
        becomes the BJD_TDB, the original stays in `mjd_nohelio`.

        LSST `mjd` is `midpointMjdTai`: already mid-exposure, in TAI.
        """
        barycentric_time_correction(astro_object, scale="tai")

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
                            & (mag_det["e_brightness"] < 1.0) #e_brightness me esta quedando entre 18 y 20 #esto era 1.0
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
