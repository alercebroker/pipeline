import numpy as np
import pandas as pd

from correction_step.core.strategy import ztf

from tests import utils


def test_ztf_strategy_corrected_is_based_on_distance():
    detections = pd.DataFrame({"distnr": np.linspace(1, 2, 10)})

    corrected = ztf.is_corrected(detections)

    assert (detections["distnr"] >= ztf.DISTANCE_THRESHOLD).any()  # Fix test
    assert (corrected == (detections["distnr"] < ztf.DISTANCE_THRESHOLD)).all()


def test_ztf_strategy_first_detection_with_close_source_splits_by_aid_and_fid():
    detections = pd.DataFrame.from_records({
        "candid": ["fn", "fy", "sy", "sn", "fy2", "fy3", "fn2", "fn3", "sy2", "sy3", "sn2", "sn3"],
        "aid": ["AID1", "AID1", "AID2", "AID2", "AID1", "AID1", "AID1", "AID1", "AID2", "AID2", "AID2", "AID2"],
        "fid": [1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2],
        "mjd": [1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3],
        "distnr": [2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    }, index="candid")

    first_corrected = ztf.is_first_corrected(detections)

    assert first_corrected[first_corrected.index.str.startswith("fy")].all()
    assert ~first_corrected[first_corrected.index.str.startswith("fn")].all()
    assert first_corrected[first_corrected.index.str.startswith("sy")].all()
    assert ~first_corrected[first_corrected.index.str.startswith("sn")].all()


def test_ztf_strategy_dubious_for_negative_difference_without_close_source():
    detections = pd.DataFrame.from_records({"distnr": [2], "isdiffpos": [-1], "aid": ["aid"], "fid": [1], "mjd": [1]})
    dubious = ztf.is_dubious(detections)
    assert dubious.all()

    detections = pd.DataFrame.from_records({"distnr": [2], "isdiffpos": [1], "aid": ["aid"], "fid": [1], "mjd": [1]})
    dubious = ztf.is_dubious(detections)
    assert ~dubious.all()


def test_ztf_strategy_dubious_for_first_with_close_source_and_follow_up_without():
    detections = pd.DataFrame.from_records(
        {"distnr": [1, 2], "isdiffpos": [1, 1], "aid": ["aid", "aid"], "fid": [1, 1], "mjd": [1, 2],
         "candid": ["a", "b"]}, index="candid"
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.loc["a"]
    assert dubious.loc["b"]

    detections = pd.DataFrame.from_records(
        {"distnr": [1, 1], "isdiffpos": [1, 1], "aid": ["aid", "aid"], "fid": [1, 1], "mjd": [1, 2],
         "candid": ["a", "b"]}, index="candid"
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.all()


def test_ztf_strategy_dubious_for_follow_up_with_close_source_and_first_without():
    detections = pd.DataFrame.from_records(
        {"distnr": [2, 1], "isdiffpos": [1, 1], "aid": ["aid", "aid"], "fid": [1, 1], "mjd": [1, 2],
         "candid": ["a", "b"]}, index="candid"
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.loc["a"]
    assert dubious.loc["b"]

    detections = pd.DataFrame.from_records(
        {"distnr": [2, 2], "isdiffpos": [1, 1], "aid": ["aid", "aid"], "fid": [1, 1], "mjd": [1, 2],
         "candid": ["a", "b"]}, index="candid"
    )
    dubious = ztf.is_dubious(detections)
    assert ~dubious.all()


def test_ztf_strategy_correction_with_low_reference_flux_equals_difference_magnitude():
    detections = pd.DataFrame.from_records(
        {"magnr": [200.], "sigmagnr": [2.], "mag": [5.], "e_mag": [.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert np.isclose(corrected["mag_corr"], 5.)
    assert np.isclose(corrected["e_mag_corr"], .1)
    assert np.isclose(corrected["e_mag_corr_ext"], .1)


def test_ztf_strategy_correction_with_low_difference_flux_equals_reference_magnitude():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.], "sigmagnr": [2.], "mag": [200.], "e_mag": [.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert np.isclose(corrected["mag_corr"], 5.)
    assert np.isclose(corrected["e_mag_corr"], np.inf)
    assert np.isclose(corrected["e_mag_corr_ext"], 0)


def test_ztf_strategy_correction_with_positive_difference_has_lower_corrected_magnitude_than_reference():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.], "sigmagnr": [2.], "mag": [5.], "e_mag": [.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert (corrected["mag_corr"] < 5.).all()


def test_ztf_strategy_correction_with_negative_difference_has_higher_corrected_magnitude_than_reference():
    detections = pd.DataFrame.from_records(
        {"magnr": [5.], "sigmagnr": [2.], "mag": [5.], "e_mag": [.1], "isdiffpos": [-1]}
    )
    corrected = ztf.correct(detections)

    assert (corrected["mag_corr"] > 5.).all()


def test_ztf_strategy_correction_with_null_nr_fields_results_in_null_corrections():
    detections = pd.DataFrame.from_records(
        {"magnr": [None], "sigmagnr": [None], "mag": [200.], "e_mag": [.1], "isdiffpos": [1]}
    )
    corrected = ztf.correct(detections)

    assert corrected["mag_corr"].isna().all()
    assert corrected["e_mag_corr"].isna().all()
    assert corrected["e_mag_corr_ext"].isna().all()
