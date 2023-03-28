from copy import deepcopy
from unittest import mock

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from correction_step.core import Corrector
from tests.utils import ztf_alert, atlas_alert

detections = [ztf_alert(candid="c1"), atlas_alert(candid="c2")]
MAG_CORR_COLS = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
ALL_NEW_COLS = MAG_CORR_COLS + ["dubious", "stellar", "corrected"]


def test_mask_survey_returns_only_alerts_from_requested_survey():
    corrector = Corrector(detections)
    expected_ztf = pd.Series([True, False], index=["c1", "c2"])
    assert (corrector._survey_mask("ZtF") == expected_ztf).all()
    assert (corrector._survey_mask("ATla") == ~expected_ztf).all()


@mock.patch("correction_step.core.corrector.strategy")
def test_apply_all_calls_requested_function_to_masked_detections_for_each_submodule(mock_strategy):
    mock_strategy.ztf = mock.MagicMock()
    mock_strategy.ztf.function = mock.MagicMock()
    mock_strategy.ztf.function.return_value = None
    mock_strategy.dummy = mock.MagicMock()
    mock_strategy.dummy.function = mock.MagicMock()

    corrector = Corrector(detections)
    corrector._apply_all_surveys("function")
    called, = mock_strategy.ztf.function.call_args.args
    assert_frame_equal(called, corrector._detections.loc[["c1"]])
    mock_strategy.dummy.function.assert_not_called()


@mock.patch("correction_step.core.corrector.strategy")
def test_apply_all_returns_a_series_with_default_value_and_dtype_if_no_columns_are_given(mock_strategy):
    corrector = Corrector(detections)
    output = corrector._apply_all_surveys("function", default=-1, dtype=float)
    assert isinstance(output, pd.Series)
    assert output.dtype == float
    assert (output.index == ["c1", "c2"]).all()
    assert (output == -1).all()


@mock.patch("correction_step.core.corrector.strategy")
def test_apply_all_returns_a_df_with_default_value_and_dtype_if_columns_are_given(mock_strategy):
    corrector = Corrector(detections)
    output = corrector._apply_all_surveys("function", default=-1, columns=["a", "b"], dtype=float)
    assert isinstance(output, pd.DataFrame)
    assert (output.dtypes == float).all()
    assert (output.columns == ["a", "b"]).all()
    assert (output.index == ["c1", "c2"]).all()
    assert (output == -1).all().all()


def test_corrected_calls_apply_all_with_function_is_corrected():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.corrected
    corrector._apply_all_surveys.assert_called_with("is_corrected", default=False, dtype=bool)


def test_corrected_is_false_for_surveys_without_strategy():
    corrector = Corrector(detections)
    assert (corrector.corrected == pd.Series([True, False], index=["c1", "c2"])).all()


def test_dubious_calls_apply_all_with_function_is_dubious():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.dubious
    corrector._apply_all_surveys.assert_called_with("is_dubious", default=False, dtype=bool)


def test_dubious_is_false_for_surveys_without_strategy():
    corrector = Corrector(detections)
    assert (corrector.dubious == pd.Series([False, False], index=["c1", "c2"])).all()


def test_stellar_calls_apply_all_with_function_is_stellar():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    _ = corrector.stellar
    corrector._apply_all_surveys.assert_called_with("is_stellar", default=False, dtype=bool)


def test_stellar_is_false_for_surveys_without_strategy():
    corrector = Corrector(detections)
    assert (corrector.stellar == pd.Series([True, False], index=["c1", "c2"])).all()


def test_correct_calls_apply_all_with_function_correct():
    corrector = Corrector(detections)
    corrector._apply_all_surveys = mock.MagicMock()
    corrector._correct()
    corrector._apply_all_surveys.assert_called_with("correct", columns=MAG_CORR_COLS, dtype=float)


def test_correct_is_nan_for_surveys_without_strategy():
    corrector = Corrector(detections)
    assert ~corrector._correct().loc["c1"].isna().any()
    assert corrector._correct().loc["c2"].isna().all()


def test_corrected_dataframe_has_generic_columns_and_new_ones_from_corrected():
    corrector = Corrector(detections)
    generic = ["aid", "oid", "tid", "fid", "mjd", "has_stamp", "isdiffpos", "mag", "e_mag"]
    new_columns = ALL_NEW_COLS
    assert (corrector.corrected_dataframe().columns.isin(generic + new_columns)).all()


def test_corrected_dataframe_sets_non_corrected_detections_to_nan():
    altered_detections = deepcopy(detections)
    altered_detections[0]["extra_fields"]["distnr"] = 2
    corrector = Corrector(altered_detections)
    assert ~corrector._correct().loc["c1"].isna().any()
    assert corrector.corrected_dataframe().loc["c1"][MAG_CORR_COLS].isna().all()


def test_corrected_dataframe_sets_infinite_values_to_zero_magnitude():
    altered_detections = deepcopy(detections)
    altered_detections[0]["isdiffpos"] = -1
    corrector = Corrector(altered_detections)
    assert np.isinf(corrector._correct().loc["c1"]).any()
    assert (corrector.corrected_dataframe().loc["c1"][MAG_CORR_COLS] == Corrector._ZERO_MAG).all()


def test_corrected_records_restores_is_same_as_original_input_with_new_corrected_fields():
    corrector = Corrector(detections)
    records = corrector.corrected_records()
    for record in records:
        for col in ALL_NEW_COLS:
            record.pop(col)
    assert records == detections
