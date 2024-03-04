from copy import deepcopy
from unittest import mock

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from correction import Corrector
from tests.utils import ztf_alert, atlas_alert

detections = [ztf_alert(candid="c1", oid="oid_ztf"), atlas_alert(candid="c2", oid="oid_atlas")]
MAG_CORR_COLS = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
ALL_NEW_COLS = MAG_CORR_COLS + ["dubious", "stellar", "corrected"]


def test_corrector_removes_duplicate_candids():
    detections_duplicate = [ztf_alert(candid="c"), atlas_alert(candid="c")]
    corrector = Corrector(detections_duplicate)
    assert (corrector._detections.index == ["c_OID1"]).all()


def test_mask_survey_returns_only_alerts_from_requested_survey():
    corrector = Corrector(detections)
    expected_ztf = pd.Series([True, False], index=["c1_oid_ztf", "c2_oid_atlas"])
    assert (corrector._survey_mask("ZtF") == expected_ztf).all()
    assert (corrector._survey_mask("ATlas") == ~expected_ztf).all()


@mock.patch("correction.core.corrector.strategy")
def test_apply_all_calls_requested_function_to_masked_detections_for_each_submodule(mock_strategy):
    mock_strategy.ztf = mock.MagicMock()
    mock_strategy.ztf.function = mock.MagicMock()
    mock_strategy.ztf.function.return_value = None
    mock_strategy.dummy = mock.MagicMock()
    mock_strategy.dummy.function = mock.MagicMock()

    corrector = Corrector(detections)
    corrector._apply_all_surveys("function")
    (called,) = mock_strategy.ztf.function.call_args.args
    assert_frame_equal(called, corrector._detections.loc[["c1_oid_ztf"]])
    mock_strategy.dummy.function.assert_not_called()


@mock.patch("correction.core.corrector.strategy")
def test_apply_all_returns_a_series_with_default_value_and_dtype_if_no_columns_are_given(
    mock_strategy,
):
    corrector = Corrector(detections)
    output = corrector._apply_all_surveys("function", default=-1, dtype=float)
    assert isinstance(output, pd.Series)
    assert output.dtype == float
    assert (output.index == ["c1_oid_ztf", "c2_oid_atlas"]).all()
    assert (output == -1).all()


@mock.patch("correction.core.corrector.strategy")
def test_apply_all_returns_a_df_with_default_value_and_dtype_if_columns_are_given(mock_strategy):
    corrector = Corrector(detections)
    output = corrector._apply_all_surveys("function", default=-1, columns=["a", "b"], dtype=float)
    assert isinstance(output, pd.DataFrame)
    assert (output.dtypes == float).all()
    assert (output.columns == ["a", "b"]).all()
    assert (output.index == ["c1_oid_ztf", "c2_oid_atlas"]).all()
    assert (output == -1).all().all()


def test_corrected_calls_apply_all_with_function_is_corrected():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.corrected
    corrector._apply_all_surveys.assert_called_with("is_corrected", default=False, dtype=bool)


def test_corrected_is_false_for_surveys_without_strategy():
    corrector = Corrector(detections)
    corrected = [x["extra_fields"].get("distnr", 99) < 1.4 for x in detections]
    assert (corrector.corrected == pd.Series(corrected, index=["c1_oid_ztf", "c2_oid_atlas"])).all()


def test_dubious_calls_apply_all_with_function_is_dubious():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.dubious
    corrector._apply_all_surveys.assert_called_with("is_dubious", default=False, dtype=bool)


def test_dubious_is_false_for_surveys_without_strategy():
    corrector = Corrector(detections)
    assert (corrector.dubious == pd.Series([False, False], index=["c1_oid_ztf", "c2_oid_atlas"])).all()


def test_stellar_calls_apply_all_with_function_is_stellar():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.stellar
    corrector._apply_all_surveys.assert_called_with("is_stellar", default=False, dtype=bool)


@mock.patch("correction.core.corrector.strategy.ztf.is_corrected")
def test_stellar_is_false_for_surveys_without_strategy(is_corrected):
    is_corrected.return_value = pd.Series([True], index=["c1_oid_ztf"])
    corrector = Corrector(detections)
    assert (corrector.stellar == pd.Series([True, False], index=["c1_oid_ztf", "c2_oid_atlas"])).all()


def test_corrected_magnitudes_calls_apply_all_with_function_correct():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    corrector.corrected_magnitudes()
    corrector._apply_all_surveys.assert_any_call("correct", columns=MAG_CORR_COLS, dtype=float)


def test_corrected_magnitudes_is_nan_for_surveys_without_strategy():
    corrector = Corrector(detections)
    if detections[0]["extra_fields"].get("distnr", 99) < 1.4:
        assert ~corrector.corrected_magnitudes().loc["c1_oid_ztf"].isna().any()
    else:
        assert corrector.corrected_magnitudes().loc["c1_oid_ztf"].isna().all()
    assert corrector.corrected_magnitudes().loc["c2_oid_atlas"].isna().all()


def test_corrected_magnitudes_sets_non_corrected_detections_to_nan():
    altered_detections = deepcopy(detections)
    altered_detections[0]["extra_fields"]["distnr"] = 2
    corrector = Corrector(altered_detections)
    assert corrector.corrected_magnitudes().loc["c1_oid_ztf"][MAG_CORR_COLS].isna().all()


def test_corrected_as_records_sets_infinite_values_to_zero_magnitude():
    altered_detections = deepcopy(detections)
    altered_detections[0]["isdiffpos"] = -1
    altered_detections[0]["extra_fields"]["distnr"] = 1  # force is_corrected to be True
    corrector = Corrector(altered_detections)
    assert all(
        corrector.corrected_as_records()[0][col] == Corrector._ZERO_MAG for col in MAG_CORR_COLS
    )


def test_corrected_as_records_restores_original_input_with_new_corrected_fields():
    corrector = Corrector(detections)
    records = corrector.corrected_as_records()
    for record in records:
        for col in ALL_NEW_COLS:
            record.pop(col)
    assert records == detections


def test_weighted_mean_with_equal_weights_is_same_as_ordinary_mean():
    vals, weights = pd.Series([100, 200]), pd.Series([5, 5])
    assert Corrector.weighted_mean(vals, weights) == 150


def test_weighted_mean_with_very_high_error_does_not_consider_its_value_in_mean():
    vals, sigmas = pd.Series([100, 200]), pd.Series([1, 1e6])
    assert np.isclose(Corrector.weighted_mean(vals, sigmas), 100)


def test_weighted_mean_with_very_small_error_only_considers_its_value_in_mean():
    vals, sigmas = pd.Series([100, 200]), pd.Series([1, 1e-6])
    assert np.isclose(Corrector.weighted_mean(vals, sigmas), 200)


def test_arcsec2deg_applies_proper_conversion():
    assert np.isclose(Corrector.arcsec2dec(1), 1 / 3600)


def test_calculate_coordinates_with_equal_weights_is_same_as_ordinary_mean():
    wdetections = [
        ztf_alert(candid="c1", ra=100, e_ra=5, forced=False),
        atlas_alert(candid="c2", ra=200, e_ra=5),
    ]
    corrector = Corrector(wdetections)
    assert corrector._calculate_coordinates("ra")["meanra"].loc["OID1"] == 150


def test_calculate_coordinates_with_a_very_high_error_does_not_consider_its_value_in_mean():
    wdetections = [
        ztf_alert(candid="c1", ra=100, e_ra=5, forced=False),
        atlas_alert(candid="c2", ra=200, e_ra=1e6),
    ]
    corrector = Corrector(wdetections)
    assert np.isclose(corrector._calculate_coordinates("ra")["meanra"].loc["OID1"], 100)


def test_calculate_coordinates_with_an_very_small_error_only_considers_its_value_in_mean():
    wdetections = [
        ztf_alert(candid="c1", ra=100, e_ra=5),
        atlas_alert(candid="c2", ra=200, e_ra=1e-6),
    ]
    corrector = Corrector(wdetections)
    assert np.isclose(corrector._calculate_coordinates("ra")["meanra"].loc["OID1"], 200)


def test_calculate_coordinates_ignores_forced_photometry():
    wdetections = [
        ztf_alert(candid="c1", ra=100, e_ra=1, forced=False),
        atlas_alert(candid="c2", ra=200, forced=True, e_ra=1),
    ]
    corrector = Corrector(wdetections)
    assert np.isclose(corrector._calculate_coordinates("ra")["meanra"].loc["OID1"], 100)


def test_coordinates_dataframe_calculates_mean_for_each_aid():
    detections_duplicate = [ztf_alert(candid="c", forced=False), atlas_alert(candid="c")]
    corrector = Corrector(detections_duplicate)
    assert corrector.mean_coordinates().index == ["OID1"]

    altered_detections = deepcopy(detections_duplicate)
    altered_detections[0]["oid"] = "OID1"
    corrector = Corrector(altered_detections)
    assert corrector.mean_coordinates().index.isin(["OID1", "OID2"]).all()


def test_coordinates_dataframe_includes_mean_ra_and_mean_dec():
    corrector = Corrector(detections)
    assert corrector.mean_coordinates().columns.isin(["meanra", "meandec"]).all()


def test_coordinates_records_has_one_entry_per_aid():
    test_detections = [ztf_alert(candid="c1", forced=False), atlas_alert(candid="c2")]
    corrector = Corrector(test_detections)
    assert set(corrector.coordinates_as_records()) == {"OID1"}

    altered_detections = deepcopy(test_detections)
    altered_detections[0]["oid"] = "OID2"
    corrector = Corrector(altered_detections)
    assert set(corrector.coordinates_as_records()) == {"OID1", "OID2"}


def test_coordinates_records_has_mean_ra_and_mean_dec_for_each_record():
    altered_detections = deepcopy(detections)
    altered_detections = [{"forced": False, **x} for x in altered_detections]
    altered_detections[0]["oid"] = "OID2"
    corrector = Corrector(altered_detections)
    for values in corrector.coordinates_as_records().values():
        assert set(values) == {"meanra", "meandec"}
