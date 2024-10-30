from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from magstats_step.core import ObjectStatistics


def test_arcsec_to_degree_conversion():
    arcsec = 3600
    degrees = ObjectStatistics._arcsec2deg(arcsec)
    assert degrees == arcsec / 3600


def test_degree_to_arcsec_conversion():
    degrees = 1
    arcsec = ObjectStatistics._deg2arcsec(degrees)
    assert degrees == arcsec / 3600


def test_composition_of_arcsec2dec_and_dec2arcsec_results_in_same_input():
    degrees = 1
    result = ObjectStatistics._arcsec2deg(
        ObjectStatistics._deg2arcsec(degrees)
    )
    assert degrees == result


def test_calculate_weights_gives_inverse_square_of_errors():
    sigs = pd.Series([2.0, 3.0])
    result = ObjectStatistics._compute_weights(sigs)
    assert (result == 1 / sigs**2).all()


def test_calculate_weighted_mean_with_equal_errors_is_standard_mean():
    vals = pd.Series([100, 200])
    sigs = pd.Series([4.0, 4.0])
    result = ObjectStatistics._weighted_mean(vals, sigs)
    assert result == 150


def test_calculate_weighted_mean_with_one_very_small_error_has_that_value_as_result():
    vals = pd.Series([100, 200])
    sigs = pd.Series([1e-6, 4.0])
    result = ObjectStatistics._weighted_mean(vals, sigs)
    assert np.isclose(result, 100)


def test_calculate_weighted_mean_with_one_very_large_error_has_that_value_disregarded():
    vals = pd.Series([100, 200])
    sigs = pd.Series([1e6, 4.0])
    result = ObjectStatistics._weighted_mean(vals, sigs)
    assert np.isclose(result, 200)


def test_calculate_weighted_mean_error_with_equal_weights_is_sigma_divided_by_sqrt_number_of_samples():
    sigs = pd.Series([4.0, 4.0])
    result = ObjectStatistics._weighted_mean_error(sigs)
    assert np.isclose(result, 4 / np.sqrt(2))


def test_calculate_weighted_mean_error_with_one_very_small_error_has_that_error_as_result():
    sigs = pd.Series([1e-6, 4.0])
    result = ObjectStatistics._weighted_mean_error(sigs)
    assert np.isclose(result, 1e-6)


def test_calculate_weighted_mean_error_with_one_very_large_error_has_that_error_disregarded():
    sigs = pd.Series([1e6, 4.0])
    result = ObjectStatistics._weighted_mean_error(sigs)
    assert np.isclose(result, 4)


def test_calculate_coordinates_with_ra_uses_weighted_mean_and_weighted_mean_error_per_oid():
    detections = [
        {"oid": "OID1", "ra": 10, "e_ra": 2, "candid": "a", "forced": False},
        {
            "oid": "OID2",
            "ra": 10,
            "ra": 20,
            "e_ra": 4,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID1",
            "ra": 10,
            "ra": 20,
            "e_ra": 4,
            "candid": "b",
            "forced": False,
        },
    ]
    calculator = ObjectStatistics(detections)

    calculator._weighted_mean = mock.Mock()
    calculator._weighted_mean.return_value = 1  # Dummy value for check
    calculator._weighted_mean_error = (
        mock.Mock()
    )  # DO NOT USE MagicMock!! Messes with some checks in pandas
    calculator._weighted_mean_error.return_value = 2  # Dummy value for check

    result = calculator._calculate_coordinates("ra")
    assert "meanra" in result and "sigmara" in result
    assert (result["meanra"] == 1).all()
    assert (result["sigmara"] == 2 * 3600).all()

    for call in calculator._weighted_mean.call_args_list:
        val, err = call.args
        assert call.kwargs == {}
        if len(val) == 2:  # This is OID1, the order cannot be assured
            assert_series_equal(
                val,
                pd.Series(
                    [10, 20],
                    index=pd.Index(["a_OID1", "b_OID1"]),
                    name="ra",
                ),
            )
            assert_series_equal(
                err,
                pd.Series(
                    [2 / 3600, 4 / 3600],
                    index=pd.Index(["a_OID1", "b_OID1"]),
                    name="e_ra",
                ),
            )
        else:  # Should be OID2
            assert_series_equal(
                val,
                pd.Series(
                    [20],
                    index=pd.Index(
                        ["c_OID2"],
                    ),
                    name="ra",
                ),
            )
            assert_series_equal(
                err,
                pd.Series(
                    [4 / 3600],
                    index=pd.Index(["c_OID2"]),
                    name="e_ra",
                ),
            )


def test_calculate_coordinates_with_dec_uses_weighted_mean_and_weighted_mean_error_per_oid():
    detections = [
        {"oid": "OID1", "dec": 10, "e_dec": 2, "candid": "a", "forced": False},
        {"oid": "OID2", "dec": 20, "e_dec": 4, "candid": "c", "forced": False},
        {"oid": "OID1", "dec": 20, "e_dec": 4, "candid": "b", "forced": False},
    ]
    calculator = ObjectStatistics(detections)

    calculator._weighted_mean = mock.Mock()
    calculator._weighted_mean.return_value = 1  # Dummy value for check
    calculator._weighted_mean_error = (
        mock.Mock()
    )  # DO NOT USE MagicMock!! Messes with some checks in pandas
    calculator._weighted_mean_error.return_value = 2  # Dummy value for check

    result = calculator._calculate_coordinates("dec")
    assert "meandec" in result and "sigmadec" in result
    assert (result["meandec"] == 1).all()
    assert (result["sigmadec"] == 2 * 3600).all()

    for call in calculator._weighted_mean.call_args_list:
        val, err = call.args
        assert call.kwargs == {}
        if len(val) == 2:
            assert_series_equal(
                val,
                pd.Series(
                    [10, 20],
                    index=pd.Index(["a_OID1", "b_OID1"]),
                    name="dec",
                ),
            )
            assert_series_equal(
                err,
                pd.Series(
                    [2 / 3600, 4 / 3600],
                    index=pd.Index(["a_OID1", "b_OID1"]),
                    name="e_dec",
                ),
            )
        else:
            assert_series_equal(
                val,
                pd.Series([20], index=pd.Index(["c_OID2"]), name="dec"),
            )
            assert_series_equal(
                err,
                pd.Series(
                    [4 / 3600],
                    index=pd.Index(["c_OID2"]),
                    name="e_dec",
                ),
            )


def test_calculate_unique_gives_list_of_unique_values_in_field_per_oid():
    detections = [
        {"oid": "OID1", "candid": "a", "extra": "A", "forced": False},
        {"oid": "OID2", "candid": "c", "extra": "A", "forced": False},
        {"oid": "OID1", "candid": "b", "extra": "A", "forced": False},
        {"oid": "OID1", "candid": "d", "extra": "B", "forced": False},
        {"oid": "OID1", "candid": "e", "extra": "B", "forced": True},
    ]
    calculator = ObjectStatistics(detections)
    result = calculator._calculate_unique("extra")

    assert "extra" in result
    assert_series_equal(
        result["extra"],
        pd.Series(
            [["A", "B"], ["A"]],
            index=pd.Index(["OID1", "OID2"], name="oid"),
            name="extra",
        ),
    )


def test_calculate_ra_uses_calculate_coordinates():
    detections = [{"oid": "OIDa", "candid": "a", "forced": False}]
    calculator = ObjectStatistics(detections)

    calculator._calculate_coordinates = mock.Mock()
    calculator.calculate_ra()

    calculator._calculate_coordinates.assert_called_once_with("ra")


def test_calculate_dec_uses_calculate_coordinates():
    detections = [{"oid": "OIDa", "candid": "a", "forced": False}]
    calculator = ObjectStatistics(detections)

    calculator._calculate_coordinates = mock.Mock()
    calculator.calculate_dec()

    calculator._calculate_coordinates.assert_called_once_with("dec")


def test_calculate_ndet_gives_number_of_detections_per_oid():
    detections = [
        {"oid": "OID1", "candid": "a", "forced": False},
        {"oid": "OID2", "candid": "c", "forced": False},
        {"oid": "OID1", "candid": "b", "forced": False},
        {"oid": "OID1", "candid": "d", "forced": False},
        {"oid": "OID1", "candid": "e", "forced": True},
    ]
    calculator = ObjectStatistics(detections)
    result = calculator.calculate_ndet()

    assert "ndet" in result
    assert_series_equal(
        result["ndet"],
        pd.Series(
            [3, 1], index=pd.Index(["OID1", "OID2"], name="oid"), name="ndet"
        ),
    )


def test_calculate_firstmjd_gives_the_first_mjd_per_oid():
    detections = [
        {"oid": "OID1", "mjd": 1, "candid": "a", "forced": False},
        {"oid": "OID2", "mjd": 2, "candid": "c", "forced": False},
        {"oid": "OID1", "mjd": 3, "candid": "b", "forced": False},
        {"oid": "OID1", "mjd": 2, "candid": "d", "forced": False},
        {"oid": "OID1", "mjd": 2, "candid": "e", "forced": True},
    ]
    calculator = ObjectStatistics(detections)
    result = calculator.calculate_firstmjd()

    assert "firstmjd" in result
    assert_series_equal(
        result["firstmjd"],
        pd.Series(
            [1, 2],
            index=pd.Index(["OID1", "OID2"], name="oid"),
            name="firstmjd",
        ),
    )


def test_calculate_lastmjd_gives_the_last_mjd_per_oid():
    detections = [
        {"oid": "OID1", "mjd": 1, "candid": "a", "forced": False},
        {"oid": "OID2", "mjd": 2, "candid": "c", "forced": False},
        {"oid": "OID1", "mjd": 3, "candid": "b", "forced": False},
        {"oid": "OID1", "mjd": 2, "candid": "d", "forced": False},
        {"oid": "OID1", "mjd": 2, "candid": "e", "forced": True},
    ]
    calculator = ObjectStatistics(detections)
    result = calculator.calculate_lastmjd()

    assert "lastmjd" in result
    assert_series_equal(
        result["lastmjd"],
        pd.Series(
            [3, 2],
            index=pd.Index(["OID1", "OID2"], name="oid"),
            name="lastmjd",
        ),
    )


def test_calculate_oid_uses_calculate_unique_with_oid():
    detections = [{"oid": "OIDa", "candid": "a", "forced": False}]
    calculator = ObjectStatistics(detections)

    calculator._calculate_unique = mock.Mock()
    calculator.calculate_oid()

    calculator._calculate_unique.assert_called_once_with("oid")


def test_calculate_tid_uses_calculate_unique_with_tid():
    detections = [{"oid": "OIDa", "candid": "a", "forced": False}]
    calculator = ObjectStatistics(detections)

    calculator._calculate_unique = mock.Mock()
    calculator.calculate_tid()

    calculator._calculate_unique.assert_called_once_with("tid")


def test_calculate_corrected_gives_whether_first_detection_in_surveys_with_correct_is_corrected():
    detections = [
        {
            "oid": "OID1",
            "sid": "MOCK_SURVEY",
            "mjd": 1,
            "corrected": False,
            "candid": "a",
            "forced": False,
        },  # Should ignore
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 2,
            "corrected": True,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 3,
            "corrected": False,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "MOCK_SURVEY",
            "mjd": 1,
            "corrected": True,
            "candid": "d",
            "forced": False,
        },  # Should ignore
        {
            "oid": "OID3",
            "sid": "SURVEY",
            "mjd": 2,
            "corrected": False,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID3",
            "sid": "SURVEY",
            "mjd": 3,
            "corrected": True,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = ObjectStatistics(detections)
    calculator._CORRECTED = ("SURVEY",)
    result = calculator.calculate_corrected()

    assert "corrected" in result
    assert_series_equal(
        result["corrected"],
        pd.Series(
            [True, False],
            index=pd.Index(["OID1", "OID3"], name="oid"),
            name="corrected",
        ),
    )


def test_calculate_stellar_gives_whether_first_detection_in_surveys_with_stellar_is_corrected():
    detections = [
        {
            "oid": "OID1",
            "sid": "MOCK_SURVEY",
            "mjd": 1,
            "stellar": False,
            "candid": "a",
            "forced": False,
        },  # Should ignore
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 2,
            "stellar": True,
            "candid": "b",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 3,
            "stellar": False,
            "candid": "c",
            "forced": False,
        },
        {
            "oid": "OID2",
            "sid": "MOCK_SURVEY",
            "mjd": 1,
            "stellar": True,
            "candid": "d",
            "forced": False,
        },  # Should ignore
        {
            "oid": "OID3",
            "sid": "SURVEY",
            "mjd": 2,
            "stellar": False,
            "candid": "e",
            "forced": False,
        },
        {
            "oid": "OID3",
            "sid": "SURVEY",
            "mjd": 3,
            "stellar": True,
            "candid": "f",
            "forced": False,
        },
    ]
    calculator = ObjectStatistics(detections)
    calculator._STELLAR = ("SURVEY",)
    result = calculator.calculate_stellar()

    assert "stellar" in result
    assert_series_equal(
        result["stellar"],
        pd.Series(
            [True, False],
            index=pd.Index(["OID1", "OID3"], name="oid"),
            name="stellar",
        ),
    )


def test_object_statistics_ignores_forced_photometry():
    detections = [
        {"oid": "OIDa", "candid": "a", "check": "this", "forced": False},
        {"oid": "OIDb", "candid": "b", "check": "that", "forced": True},
    ]
    calculator = ObjectStatistics(detections)

    assert_series_equal(
        calculator._detections["check"],
        pd.Series(["this"], index=pd.Index(["a_OIDa"]), name="check"),
    )


def test_object_statistics_deltajd():
    detections = [
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 1,
            "stellar": False,
            "candid": "a",
            "forced": False,
        },
        {
            "oid": "OID1",
            "sid": "SURVEY",
            "mjd": 5,
            "stellar": True,
            "candid": "b",
            "forced": False,
        },
    ]
    calculator = ObjectStatistics(detections)
    result = calculator.calculate_deltajd()
    assert_series_equal(
        result["deltajd"],
        pd.Series([4], index=pd.Index(["OID1"], name="oid"), name="deltajd"),
    )


def test_generate_stats():
    detections = [
        {
            "oid": "OID1",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 1,
            "stellar": False,
            "candid": "a",
            "forced": False,
            "corrected": True,
            "e_ra": 1,
            "ra": 10,
            "e_dec": 1,
            "dec": 10,
            "isdiffpos": 1,
            "mjdendref": 60470.5,
        },
        {
            "oid": "OID1",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 1,
            "stellar": False,
            "candid": "a",
            "forced": False,
            "corrected": True,
            "e_ra": 1,
            "ra": 10,
            "e_dec": 1,
            "dec": 10,
            "isdiffpos": 1,
            "mjdendref": 60470.5,
        },
        {
            "oid": "OID2",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 1,
            "stellar": False,
            "candid": "a",
            "forced": False,
            "corrected": True,
            "e_ra": 1,
            "ra": 10,
            "e_dec": 1,
            "dec": 10,
            "isdiffpos": 1,
            "mjdendref": 60470.5,
        },
        {
            "oid": "OID1",
            "sid": "ZTF",
            "tid": "ZTF",
            "mjd": 5,
            "stellar": True,
            "candid": "b",
            "forced": False,
            "corrected": False,
            "e_ra": 1,
            "ra": 10,
            "e_dec": 1,
            "dec": 10,
            "isdiffpos": 1,
            "mjdendref": 60470.5,
        },
    ]
    calculator = ObjectStatistics(detections)
    result = calculator.generate_statistics()
    assert result["ndet"].loc["OID1"] == 2
    assert result["ndet"].loc["OID2"] == 1
