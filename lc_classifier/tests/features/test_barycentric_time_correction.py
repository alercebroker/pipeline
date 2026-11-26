import unittest

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

from lc_classifier.features.core.base import AstroObject
from lc_classifier.features.preprocess.barycentric import to_bjd_tdb
from lc_classifier.features.preprocess.lsst import LSSTLightcurvePreprocessor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

SECOND = 1.0 / 86400.0
# Mid-exposure instant used across tests, in UTC (inside the IERS-B table that
# ships with astropy, so the reference below needs no download).
INSTANT_UTC = 60100.3
TAI_MINUS_UTC = 37.0 * SECOND  # constant since 2017


def make_object(mjd, ra, dec, exptime=None, forced=None):
    n = len(mjd)
    detections = pd.DataFrame(
        {
            "oid": ["obj"] * n,
            "sid": ["ztf"] * n,
            "fid": ["g"] * n,
            "mjd": np.asarray(mjd, dtype=float),
            "ra": ra,
            "dec": dec,
        }
    )
    if exptime is not None:
        detections["exptime"] = exptime
    return AstroObject(
        metadata=pd.DataFrame(columns=["name", "value"]),
        detections=detections,
        forced_photometry=forced,
    )


def reference_bjd_tdb(mid_exposure_mjd, scale, ra, dec):
    """BJD_TDB from astropy's own light_travel_time, at the Earth's centre."""
    geocentre = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
    times = Time(mid_exposure_mjd, format="mjd", scale=scale, location=geocentre)
    source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    with iers.conf.set_temp("auto_download", False):
        ltt = times.light_travel_time(source, kind="barycentric")
    return (times.tdb + ltt).mjd


class TestBarycentricTimeCorrection(unittest.TestCase):
    def test_ra_wrap_matches_ra_zero(self):
        mjd = [INSTANT_UTC, INSTANT_UTC + 100.0, INSTANT_UTC + 200.0]
        wrapped = make_object(mjd, ra=[359.9999, 0.0001, 359.9999], dec=10.0)
        at_zero = make_object(mjd, ra=0.0, dec=10.0)
        preprocessor = ZTFLightcurvePreprocessor()
        preprocessor._helio_time_correction(wrapped)
        preprocessor._helio_time_correction(at_zero)

        diff_s = (wrapped.detections["mjd"] - at_zero.detections["mjd"]) / SECOND
        self.assertLess(np.max(np.abs(diff_s)), 1e-3)

    def test_ztf_moves_start_to_mid_exposure(self):
        # One object per exposure time, all starting at the same instant
        # (rows of one object sharing a start time would be one exposure).
        direction = _unit_vector(120.0, -30.0)
        corrected = []
        for exptime, half_exposure_s in [(240.0, 120.0), (30.0, 15.0), (np.nan, 15.0)]:
            ao = make_object([INSTANT_UTC], ra=120.0, dec=-30.0, exptime=[exptime])
            ZTFLightcurvePreprocessor()._helio_time_correction(ao)
            expected = to_bjd_tdb(
                np.array([INSTANT_UTC + half_exposure_s * SECOND]), direction, "utc"
            )[0]
            got = ao.detections["mjd"].iloc[0]
            self.assertLess(abs(got - expected) / SECOND, 1e-4)
            self.assertEqual(ao.detections["mjd_nohelio"].iloc[0], INSTANT_UTC)
            corrected.append(got)

        # The 240 s exposure lands 105 s after the 30 s one, give or take the
        # Earth's motion during those 105 s (~10 ms).
        gap_s = (corrected[0] - corrected[1]) / SECOND
        self.assertAlmostEqual(gap_s, 105.0, delta=0.02)

    def test_missing_exptime_borrowed_from_same_exposure(self):
        forced = pd.DataFrame(
            {
                "oid": ["obj"],
                "sid": ["ztf"],
                "fid": ["g"],
                "mjd": [INSTANT_UTC],
                "ra": [120.0],
                "dec": [-30.0],
                "exptime": [240.0],
            }
        )
        ao = make_object([INSTANT_UTC], ra=120.0, dec=-30.0, forced=forced)
        ZTFLightcurvePreprocessor()._helio_time_correction(ao)
        self.assertEqual(
            ao.detections["mjd"].iloc[0], ao.forced_photometry["mjd"].iloc[0]
        )

    def test_ztf_and_lsst_agree_on_the_same_instant(self):
        ra, dec = 210.0, 45.0
        ztf = make_object([INSTANT_UTC - 15.0 * SECOND], ra=ra, dec=dec, exptime=[30.0])
        lsst = make_object([INSTANT_UTC + TAI_MINUS_UTC], ra=ra, dec=dec)
        ZTFLightcurvePreprocessor()._helio_time_correction(ztf)
        LSSTLightcurvePreprocessor()._helio_time_correction(lsst)

        diff_s = (
            ztf.detections["mjd"].iloc[0] - lsst.detections["mjd"].iloc[0]
        ) / SECOND
        self.assertLess(abs(diff_s), 1e-3)

    def test_matches_astropy_reference(self):
        ra, dec = 300.0, 5.0
        reference = reference_bjd_tdb(np.array([INSTANT_UTC]), "utc", ra, dec)[0]

        ztf = make_object([INSTANT_UTC - 15.0 * SECOND], ra=ra, dec=dec, exptime=[30.0])
        lsst = make_object([INSTANT_UTC + TAI_MINUS_UTC], ra=ra, dec=dec)
        ZTFLightcurvePreprocessor()._helio_time_correction(ztf)
        LSSTLightcurvePreprocessor()._helio_time_correction(lsst)

        for ao in (ztf, lsst):
            error_s = (ao.detections["mjd"].iloc[0] - reference) / SECOND
            self.assertLess(abs(error_s), 0.03)


def _unit_vector(ra_deg, dec_deg):
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


if __name__ == "__main__":
    unittest.main()
