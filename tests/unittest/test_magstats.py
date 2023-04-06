from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from magstats_step.core import MagnitudeStatistics


def test_calculate_uncorrected_stats_gives_statistics_for_magnitudes_per_aid_and_fid():
    detections = [
        {"aid": "AID1", "fid": 1, "mag": 2, "candid": "a"},
        {"aid": "AID1", "fid": 1, "mag": 2, "candid": "b"},
        {"aid": "AID1", "fid": 1, "mag": 5, "candid": "c"},
        {"aid": "AID2", "fid": 1, "mag": 1, "candid": "d"},
        {"aid": "AID1", "fid": 2, "mag": 1, "candid": "e"},
        {"aid": "AID1", "fid": 2, "mag": 2, "candid": "f"},
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats(False)

    expected = pd.DataFrame({
        "magmean": [3, 1, 1.5],
        "magmedian": [2, 1, 1.5],
        "magmax": [5, 1, 2],
        "magmin": [2, 1, 1],
        "magsigma": [np.sqrt(2), 0, 0.5],
        "aid": ["AID1", "AID2", "AID1"],
        "fid": [1, 1, 2]
    })
    assert_frame_equal(result, expected.set_index(["aid", "fid"]), check_like=True)


def test_calculate_corrected_stats_gives_statistics_for_corrected_magnitudes_per_aid_and_fid():
    detections = [
        {"aid": "AID1", "fid": 1, "mag_corr": 2, "corrected": True, "candid": "a"},
        {"aid": "AID1", "fid": 1, "mag_corr": 2, "corrected": True, "candid": "b"},
        {"aid": "AID1", "fid": 1, "mag_corr": 5, "corrected": True, "candid": "c"},
        {"aid": "AID1", "fid": 1, "mag_corr": 5, "corrected": False, "candid": "c1"},
        {"aid": "AID2", "fid": 1, "mag_corr": 1, "corrected": True, "candid": "d"},
        {"aid": "AID1", "fid": 2, "mag_corr": 1, "corrected": True, "candid": "e"},
        {"aid": "AID1", "fid": 2, "mag_corr": 2, "corrected": True, "candid": "f"},
        {"aid": "AID2", "fid": 2, "mag_corr": 2, "corrected": False, "candid": "f1"},
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats(True)

    expected = pd.DataFrame({
        "magmean_corr": [3, 1, 1.5],
        "magmedian_corr": [2, 1, 1.5],
        "magmax_corr": [5, 1, 2],
        "magmin_corr": [2, 1, 1],
        "magsigma_corr": [np.sqrt(2), 0, 0.5],
        "aid": ["AID1", "AID2", "AID1"],
        "fid": [1, 1, 2]
    })
    assert_frame_equal(result, expected.set_index(["aid", "fid"]), check_like=True)


def test_calculate_uncorrected_stats_over_time_gives_first_and_last_magnitude_per_aid_and_fid():
    detections = [
        {"aid": "AID1", "fid": 1, "mjd": 3, "mag": 1, "candid": "a"},  # last
        {"aid": "AID1", "fid": 1, "mjd": 1, "mag": 2, "candid": "b"},  # first
        {"aid": "AID1", "fid": 1, "mjd": 2, "mag": 3, "candid": "c"},
        {"aid": "AID2", "fid": 1, "mjd": 1, "mag": 1, "candid": "d"},  # last and first
        {"aid": "AID1", "fid": 2, "mjd": 1, "mag": 1, "candid": "e"},  # first
        {"aid": "AID1", "fid": 2, "mjd": 2, "mag": 2, "candid": "f"},  # last
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats_over_time(False)

    expected = pd.DataFrame({
        "magfirst": [2, 1, 1],
        "maglast": [1, 1, 2],
        "aid": ["AID1", "AID2", "AID1"],
        "fid": [1, 1, 2]
    })
    assert_frame_equal(result, expected.set_index(["aid", "fid"]), check_like=True)


def test_calculate_corrected_stats_over_time_gives_first_and_last_corrected_magnitude_per_aid_and_fid():
    detections = [
        {"aid": "AID1", "fid": 1, "mjd": 3, "mag_corr": 1, "corrected": True, "candid": "a"},
        {"aid": "AID1", "fid": 1, "mjd": 1, "mag_corr": 2, "corrected": True, "candid": "b"},
        {"aid": "AID1", "fid": 1, "mjd": 2, "mag_corr": 3, "corrected": True, "candid": "c"},
        {"aid": "AID1", "fid": 1, "mjd": 4, "mag_corr": 3, "corrected": False, "candid": "c1"},
        {"aid": "AID2", "fid": 1, "mjd": 1, "mag_corr": 1, "corrected": True, "candid": "d"},
        {"aid": "AID1", "fid": 2, "mjd": 1, "mag_corr": 1, "corrected": True, "candid": "e"},
        {"aid": "AID1", "fid": 2, "mjd": 2, "mag_corr": 2, "corrected": True, "candid": "f"},
        {"aid": "AID2", "fid": 2, "mjd": 0, "mag_corr": 2, "corrected": False, "candid": "f1"},
    ]
    calculator = MagnitudeStatistics(detections)
    result = calculator._calculate_stats_over_time(True)

    expected = pd.DataFrame({
        "magfirst_corr": [2, 1, 1],
        "maglast_corr": [1, 1, 2],
        "aid": ["AID1", "AID2", "AID1"],
        "fid": [1, 1, 2]
    })
    assert_frame_equal(result, expected.set_index(["aid", "fid"]), check_like=True)
