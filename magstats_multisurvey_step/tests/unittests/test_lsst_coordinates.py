"""LSST raErr/decErr are in degrees; sigmara/sigmadec must be stored in arcsec."""

import numpy as np

from magstats_multisurvey_step.core.surveys.LSST.ObjstatsLsst import (
    LSSTObjectStatistics,
)
from magstats_multisurvey_step.core.surveys.LSST.pre_execute import (
    DEFAULT_ERR_DEG,
    extract_lsst_data,
)


def _source(measurement_id, ra, dec, ra_err, dec_err, mjd):
    return {
        "oid": 1,
        "measurement_id": measurement_id,
        "new": True,
        "ra": ra,
        "dec": dec,
        "raErr": ra_err,
        "decErr": dec_err,
        "mjd": mjd,
    }


def test_extract_lsst_data_converts_errors_to_arcsec():
    msg = {
        "sources": [_source(1, 10.0, 20.0, 1e-5, 2e-5, 60000.0)],
        "previous_sources": [_source(2, 10.0, 20.0, None, None, 59999.0)],
        "ss_sources": [_source(3, 10.0, 20.0, 3e-5, 4e-5, 59998.0)],
    }
    sources, _, _, ss_sources = extract_lsst_data(msg)

    assert np.isclose(sources[0]["ra_error"], 1e-5 * 3600)
    assert np.isclose(sources[0]["dec_error"], 2e-5 * 3600)
    assert np.isclose(sources[1]["ra_error"], DEFAULT_ERR_DEG * 3600)
    assert np.isclose(sources[1]["dec_error"], DEFAULT_ERR_DEG * 3600)
    assert np.isclose(ss_sources[0]["ra_error"], 3e-5 * 3600)
    assert np.isclose(ss_sources[0]["dec_error"], 4e-5 * 3600)


def test_lsst_sigma_coordinates_in_arcsec():
    err_deg = 1e-5
    msg = {
        "sources": [
            _source(1, 10.0, 20.0, err_deg, err_deg, 60000.0),
            _source(2, 10.0002, 20.0002, err_deg, err_deg, 60001.0),
        ],
    }
    sources, _, _, _ = extract_lsst_data(msg)
    stats = LSSTObjectStatistics(sources, survey="lsst")

    ra = stats.calculate_ra()
    dec = stats.calculate_dec()

    # Two equal errors: combined error is err / sqrt(2), expressed in arcsec
    expected_sigma = err_deg * 3600 / np.sqrt(2)
    assert np.isclose(ra.loc[1, "sigmara"], expected_sigma)
    assert np.isclose(dec.loc[1, "sigmadec"], expected_sigma)
    assert np.isclose(ra.loc[1, "meanra"], 10.0001)
    assert np.isclose(dec.loc[1, "meandec"], 20.0001)
