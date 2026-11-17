import math

import pytest

from hips_dummy_step.positions import alert_id, alert_position, clean_position


def test_ztf_position_and_id():
    message = {"candid": 7, "candidate": {"candid": 7, "ra": 10.5, "dec": -3}}
    assert alert_position(message, "ztf") == (10.5, -3)
    assert alert_id(message, "ztf") == 7


def test_lsst_position_and_id():
    message = {"diaSourceId": 9, "diaSource": {"ra": 200.0, "dec": 45.0}}
    assert alert_position(message, "lsst") == (200.0, 45.0)
    assert alert_id(message, "lsst") == 9


@pytest.mark.parametrize(
    "ra, dec, expected",
    [
        (10.0, 20.0, (10.0, 20.0)),
        (0.0, -90.0, (0.0, -90.0)),
        (360.0, 90.0, (0.0, 90.0)),
        (-10.0, 0.0, (350.0, 0.0)),
        (-1e-17, 0.0, (0.0, 0.0)),
        ("12.5", "1", (12.5, 1.0)),
    ],
)
def test_clean_position_keeps_or_wraps(ra, dec, expected):
    assert clean_position(ra, dec) == expected


@pytest.mark.parametrize(
    "ra, dec",
    [
        (None, 0.0),
        (10.0, None),
        (math.nan, 0.0),
        (10.0, math.inf),
        (10.0, 90.5),
        (10.0, -91.0),
        ("abc", 0.0),
    ],
)
def test_clean_position_rejects(ra, dec):
    assert clean_position(ra, dec) is None
