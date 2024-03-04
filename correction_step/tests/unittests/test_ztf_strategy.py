from unittest import mock

import numpy as np
import pandas as pd

from correction.core.strategy import ztf


def test_ztf_strategy_corrected_is_based_on_distance():
    detections = pd.DataFrame({"distnr": np.linspace(1, 2, 10)})

    corrected = ztf.is_corrected(detections)

    assert (detections["distnr"] >= ztf.DISTANCE_THRESHOLD).any()
    assert (corrected == (detections["distnr"] < ztf.DISTANCE_THRESHOLD)).all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
def test_ztf_strategy_first_detection_with_close_source_splits_by_oid_and_fid(mock_corrected):
    candids = ["fn", "fy", "sy", "sn", "fy2", "fy3", "fn2", "fn3", "sy2", "sy3", "sn2", "sn3"]
    mock_corrected.return_value = pd.Series(
        [False, True, True, False, True, False, True, False, True, False, True, False],
        index=candids,
    )
    detections = pd.DataFrame.from_records(
        {
            "candid": candids,
            "oid": [
                "OID1",
                "OID1",
                "OID2",
                "OID2",
                "OID1",
                "OID1",
                "OID1",
                "OID1",
                "OID2",
                "OID2",
                "OID2",
                "OID2",
            ],
            "fid": [1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2],
            "mjd": [1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3],
        },
        index="candid",
    )
    corrected = ztf.is_corrected(detections)
    first_corrected = ztf.is_first_corrected(detections, corrected)

    assert first_corrected[first_corrected.index.str.startswith("fy")].all()
    assert ~first_corrected[first_corrected.index.str.startswith("fn")].all()
    assert first_corrected[first_corrected.index.str.startswith("sy")].all()
    assert ~first_corrected[first_corrected.index.str.startswith("sn")].all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
@mock.patch("correction.core.strategy.ztf.is_first_corrected")
def test_ztf_strategy_dubious_for_negative_difference_without_close_source(
    mock_first, mock_corrected
):
    mock_corrected.return_value = pd.Series([False, True])
    mock_first.return_value = pd.Series([True, True])
    detections = pd.DataFrame.from_records(
        {
            "isdiffpos": [-1, -1],
            "distnr": [2, 2],
            "oid": ["OID1", "OID1"],
            "fid": [1, 1],
            "mjd": [1, 2],
        }
    )
    dubious = ztf.is_dubious(detections)
    assert (dubious == pd.Series([True, False])).all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
@mock.patch("correction.core.strategy.ztf.is_first_corrected")
def test_ztf_strategy_dubious_true_for_follow_up_without_close_source_and_first_with(
    mock_first, mock_corrected
):
    mock_corrected.return_value = pd.Series([True, False])
    mock_first.return_value = pd.Series([True, True])
    detections = pd.DataFrame.from_records(
        {
            "isdiffpos": [1, 1],
            "distnr": [2, 2],
            "oid": ["OID1", "OID1"],
            "fid": [1, 1],
            "mjd": [1, 2],
        }
    )
    dubious = ztf.is_dubious(detections)
    assert (dubious == pd.Series([False, True])).all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
@mock.patch("correction.core.strategy.ztf.is_first_corrected")
def test_ztf_strategy_dubious_true_for_follow_up_with_close_source_and_first_without(
    mock_first, mock_corrected
):
    mock_corrected.return_value = pd.Series([False, True])
    mock_first.return_value = pd.Series([False, False])
    detections = pd.DataFrame.from_records(
        {
            "isdiffpos": [1, 1],
            "distnr": [2, 2],
            "oid": ["OID1", "OID1"],
            "fid": [1, 1],
            "mjd": [1, 2],
        }
    )
    dubious = ztf.is_dubious(detections)
    assert (dubious == pd.Series([False, True])).all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
@mock.patch("correction.core.strategy.ztf.is_first_corrected")
def test_ztf_strategy_dubious_false_for_follow_up_without_close_source_and_first_without(
    mock_first, mock_corrected
):
    mock_corrected.return_value = pd.Series([False, False])
    mock_first.return_value = pd.Series([False, False])
    detections = pd.DataFrame.from_records(
        {
            "isdiffpos": [1, 1],
            "distnr": [2, 2],
            "oid": ["OID1", "OID1"],
            "fid": [1, 1],
            "mjd": [1, 2],
        }
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.all()


@mock.patch("correction.core.strategy.ztf.is_corrected")
@mock.patch("correction.core.strategy.ztf.is_first_corrected")
def test_ztf_strategy_dubious_false_for_follow_up_with_close_source_and_first_with(
    mock_first, mock_corrected
):
    mock_corrected.return_value = pd.Series([True, True])
    mock_first.return_value = pd.Series([True, True])
    detections = pd.DataFrame.from_records(
        {
            "isdiffpos": [1, 1],
            "distnr": [2, 2],
            "oid": ["OID1", "OID1"],
            "fid": [1, 1],
            "mjd": [1, 2],
        }
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.all()


def test_ztf_strategy_correction_with_low_reference_flux_equals_difference_magnitude():
    detections = pd.DataFrame.from_records(
        {"magnr": [200.0], "sigmagnr": [2.0], "mag": [5.0], "e_mag": [0.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert np.isclose(corrected["mag_corr"], 5.0)
    assert np.isclose(corrected["e_mag_corr"], 0.1)
    assert np.isclose(corrected["e_mag_corr_ext"], 0.1)


def test_ztf_strategy_correction_with_low_difference_flux_equals_reference_magnitude():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.0], "sigmagnr": [2.0], "mag": [200.0], "e_mag": [0.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert np.isclose(corrected["mag_corr"], 5.0)
    assert np.isclose(corrected["e_mag_corr"], np.inf)
    assert np.isclose(corrected["e_mag_corr_ext"], 0)


def test_ztf_strategy_correction_with_positive_difference_has_lower_corrected_magnitude_than_reference():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.0], "sigmagnr": [2.0], "mag": [5.0], "e_mag": [0.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert (corrected["mag_corr"] < 5.0).all()


def test_ztf_strategy_correction_with_negative_difference_has_higher_corrected_magnitude_than_reference():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.0], "sigmagnr": [2.0], "mag": [5.0], "e_mag": [0.1], "isdiffpos": [-1]}
    )
    corrected = ztf.correct(detections)

    assert (corrected["mag_corr"] > 5.0).all()


def test_ztf_strategy_correction_with_null_nr_fields_results_in_null_corrections():
    detections = pd.DataFrame.from_records(
        {"magnr": [None], "sigmagnr": [None], "mag": [200.0], "e_mag": [0.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert corrected["mag_corr"].isna().all()
    assert corrected["e_mag_corr"].isna().all()
    assert corrected["e_mag_corr_ext"].isna().all()


def test_ztf_strategy_correction_with_zeromag():
    detections = pd.DataFrame.from_records(
        {
            "magnr": [13, 15],
            "sigmagnr": [0.1, 1],
            "mag": [100, 100],
            "e_mag": [0.1, 0.2],
            "isdiffpos": [1, 0],
        }
    )
    corrected = ztf.correct(detections)
    assert np.isinf(corrected["mag_corr"]).all()
    assert np.isinf(corrected["e_mag_corr"]).all()
    assert np.isinf(corrected["e_mag_corr_ext"]).all()


def test_ztf_strategy_correction_with_zeromag_only_emag():
    detections = pd.DataFrame.from_records(
        {
            "magnr": [13, 15],
            "sigmagnr": [0.1, 1],
            "mag": [14, 15],
            "e_mag": [100, 100],
            "isdiffpos": [1, 0],
        }
    )
    corrected = ztf.correct(detections)
    assert not np.isinf(corrected["mag_corr"]).all()
    assert np.isinf(corrected["e_mag_corr"]).all()
    assert np.isinf(corrected["e_mag_corr_ext"]).all()
