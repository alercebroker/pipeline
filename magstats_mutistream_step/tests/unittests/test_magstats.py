from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from magstats_step.core import MagnitudeStatistics


def test_calculate_uncorrected_stats_gives_statistics_for_magnitudes_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag": 2,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag": 2,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag": 5,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mag": 1,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mag": 1,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mag": 2,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats(False)

    expected = pd.DataFrame(
        {
            "magmean": [3, 1, 1.5],
            "magmedian": [2, 1, 1.5],
            "magmax": [5, 1, 2],
            "magmin": [2, 1, 1],
            "magsigma": [np.sqrt(2), 0, 0.5],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_corrected_stats_gives_statistics_for_corrected_magnitudes_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag_corr": 2,
            "corrected": True,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag_corr": 2,
            "corrected": True,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag_corr": 5,
            "corrected": True,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mag_corr": 5,
            "corrected": False,
            "candid": "c1",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mag_corr": 1,
            "corrected": True,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mag_corr": 1,
            "corrected": True,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mag_corr": 2,
            "corrected": True,
            "candid": "f",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 2,
            "mag_corr": 2,
            "corrected": False,
            "candid": "f1",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats(True)

    expected = pd.DataFrame(
        {
            "magmean_corr": [3, 1, 1.5],
            "magmedian_corr": [2, 1, 1.5],
            "magmax_corr": [5, 1, 2],
            "magmin_corr": [2, 1, 1],
            "magsigma_corr": [np.sqrt(2), 0, 0.5],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_uncorrected_stats_over_time_gives_first_and_last_magnitude_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "mag": 1,
            "candid": "a",
            "forced": False,
        },  # last
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "mag": 2,
            "candid": "b",
            "forced": False,
        },  # first
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "mag": 3,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "mag": 1,
            "candid": "d",
            "forced": False,
        },  # last and first
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "mag": 1,
            "candid": "e",
            "forced": False,
        },  # first
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "mag": 2,
            "candid": "f",
            "forced": False,
        },  # last
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats_over_time(False)

    expected = pd.DataFrame(
        {
            "magfirst": [2, 1, 1],
            "maglast": [1, 1, 2],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_corrected_stats_over_time_gives_first_and_last_corrected_magnitude_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "mag_corr": 1,
            "corrected": True,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "mag_corr": 2,
            "corrected": True,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "mag_corr": 3,
            "corrected": True,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 4,
            "mag_corr": 3,
            "corrected": False,
            "candid": "c1",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "mag_corr": 1,
            "corrected": True,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "mag_corr": 1,
            "corrected": True,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "mag_corr": 2,
            "corrected": True,
            "candid": "f",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 0,
            "mag_corr": 2,
            "corrected": False,
            "candid": "f1",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats_over_time(True)

    expected = pd.DataFrame(
        {
            "magfirst_corr": [2, 1, 1],
            "maglast_corr": [1, 1, 2],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_statistics_calls_stats_and_stats_over_time_with_both_corrected_and_full_magnitudes():
    detections = [{"oid": "OIDa", "candid": "a", "forced": False}]
    calculator = MagnitudeStatistics(detections)

    calculator._calculate_stats = mock.Mock()
    calculator._calculate_stats_over_time = mock.Mock()
    calculator.calculate_statistics()

    calculator._calculate_stats.assert_any_call(corrected=True)
    calculator._calculate_stats.assert_any_call(corrected=False)
    calculator._calculate_stats_over_time.assert_any_call(corrected=True)
    calculator._calculate_stats_over_time.assert_any_call(corrected=False)


def test_calculate_firstmjd_gives_first_date_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 0,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 0.5,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_firstmjd()

    expected = pd.DataFrame(
        {
            "firstmjd": [0, 0.5, 1],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_lastmjd_gives_last_date_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_lastmjd()

    expected = pd.DataFrame(
        {
            "lastmjd": [3, 1, 2],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_corrected_gives_whether_first_detection_per_oid_and_fid_is_corrected():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "corrected": True,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "corrected": False,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "corrected": True,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "corrected": True,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "corrected": True,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "corrected": False,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_corrected()

    expected = pd.DataFrame(
        {
            "corrected": [False, True, True],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_stellar_gives_whether_first_detection_per_oid_and_fid_is_stellar():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 3,
            "stellar": True,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "stellar": False,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 2,
            "stellar": True,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "mjd": 1,
            "stellar": True,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 1,
            "stellar": True,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "mjd": 2,
            "stellar": False,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_stellar()

    expected = pd.DataFrame(
        {
            "stellar": [False, True, True],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_ndet_gives_number_of_detections_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_ndet()

    expected = pd.DataFrame(
        {
            "ndet": [3, 1, 2],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_ndubious_gives_number_of_dubious_detections_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "dubious": True,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "dubious": True,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 1,
            "dubious": False,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "SURVEY",
            "fid": 1,
            "dubious": False,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "dubious": True,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 2,
            "dubious": False,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_ndubious()

    expected = pd.DataFrame(
        {
            "ndubious": [2, 0, 1],
            "oid": ["OID1", "OID2", "OID1"],
            "sid": ["SURVEY", "SURVEY", "SURVEY"],
            "fid": [1, 1, 2],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_calculate_saturation_rate_gives_saturation_ratio_per_oid_and_fid():
    detections = [
        {
            "oid": "OID1",
            "sid": "ZTF",
            "fid": 1,
            "corrected": True,
            "mag_corr": 0,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "ZTF",
            "fid": 1,
            "corrected": True,
            "mag_corr": 100,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "ZTF",
            "fid": 1,
            "corrected": True,
            "mag_corr": 100,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "ZTF",
            "fid": 1,
            "corrected": True,
            "mag_corr": 0,
            "candid": "c1",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "ZTF",
            "fid": 2,
            "corrected": False,
            "mag_corr": np.nan,
            "candid": "d",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "ZTF",
            "fid": 3,
            "corrected": False,
            "mag_corr": np.nan,
            "candid": "d1",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "ZTF",
            "fid": 3,
            "corrected": True,
            "mag_corr": 100,
            "candid": "d2",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 10,
            "corrected": True,
            "mag_corr": 0,
            "candid": "e",
            "forced": False,
        },  # No threshold
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "fid": 10,
            "corrected": True,
            "mag_corr": 100,
            "candid": "f",
            "forced": False,
        },  # No threshold
        {
            "oid": "OID3",
            "sid": "SURVEY",
            "fid": 1,
            "corrected": False,
            "mag_corr": 0,
            "candid": "g",
            "forced": False,
        },  # No threshold
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator.calculate_saturation_rate()

    expected = pd.DataFrame(
        {
            "saturation_rate": [0.5, np.nan, 0, np.nan, np.nan],
            "oid": ["OID1", "OID2", "OID2", "OID1", "OID3"],
            "sid": ["ZTF", "ZTF", "ZTF", "SURVEY", "SURVEY"],
            "fid": [1, 2, 3, 10, 1],
        }
    )
    assert_frame_equal(
        result, expected.set_index(["oid", "sid", "fid"]), check_like=True
    )


def test_magnitude_statistics_ignores_forced_photometry():
    detections = [
        {"oid": "OIDa", "candid": "a", "forced": False},
        {"oid": "OIDb", "candid": "b", "forced": True},
    ]
    calculator = MagnitudeStatistics(detections)

    assert_frame_equal(
        calculator._detections,
        pd.DataFrame(
            {"oid": "OIDa", "candid": "a", "forced": False},
            index=pd.Index(["a_OIDa"]),
        ),
    )
