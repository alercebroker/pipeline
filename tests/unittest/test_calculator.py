import math
import pandas as pd
from unittest import mock
from magstats_step.core.utils.compose import compose as super_magstats_calculator
from magstats_step.core.utils.magstats_intersection import (
    CALCULATORS_LIST,
    magstats_intersection,
)
from magstats_step.core.calculators import *
from data.messages import data
from data.utils import setup_calculator_args


def test_magstats_intersection():
    excluded_calcs = list(set(CALCULATORS_LIST) - set(["ra", "dec"]))
    result = magstats_intersection(excluded_calcs)
    expected_result = [calculate_ra, calculate_dec]
    assert set(result.values()) == set(expected_result)


def test_super_magstats_calculator():
    ra_calculator = mock.MagicMock()
    dec_calculator = mock.MagicMock()
    functions = [ra_calculator, dec_calculator]
    super_magstats_calculator(*functions)([data[0]])
    ra_calculator.assert_called_with(dec_calculator(data[0]))


def test_magstats_calculators_composition():
    excluded_calcs = list(set(CALCULATORS_LIST) - set(["ra", "dec"]))
    calculators = magstats_intersection(excluded_calcs)
    obj, det, nondet = setup_calculator_args(data[0])
    alerce_object, _, __ = super_magstats_calculator(*calculators.values())(obj, det, nondet)

    assert alerce_object.meanra is not None
    assert alerce_object.sigmara is not None
    assert alerce_object.meandec is not None
    assert alerce_object.sigmadec is not None


def test_calculate_stats_coordinates():
    coords = pd.Series([250, 250, 250])
    e_coords = pd.Series([3600, 3600, 3600])
    expected_result = (250, math.sqrt(1 / 3) * 3600)
    result = calculate_stats_coordinates(coords, e_coords)
    assert result == expected_result


def test_calculate_stats_with_equal_weights():
    coords = pd.Series([100, 150, 200])
    e_coords = pd.Series([1, 1, 1])
    expected_result = (150, math.sqrt(1 / 3))
    result = calculate_stats_coordinates(coords, e_coords)
    assert math.isclose(result[0], expected_result[0])
    assert math.isclose(result[1], expected_result[1])


def test_calculate_dec():
    obj, det, nondet = setup_calculator_args(data[0])
    alerce_object = calculate_dec(obj, det, nondet)[0]
    assert alerce_object.meandec is not None
    assert alerce_object.sigmadec is not None


def test_calculate_ra():
    obj, det, nondet = setup_calculator_args(data[0])

    alerce_object = calculate_ra(obj, det, nondet)[0]
    assert alerce_object.meanra is not None
    assert alerce_object.sigmara is not None


def test_calculate_mjd():
    obj, det, nondet = setup_calculator_args(data[0])
    alerce_object = calculate_mjd(obj, det, nondet)[0]

    assert alerce_object.firstmjd < alerce_object.lastmjd


def test_calculate_ndet():
    args = setup_calculator_args(data[0])
    alerce_object = calculate_ndet(*args)[0]

    assert alerce_object.ndet is not None


def test_calculate_magnitude_statistics():
    args = setup_calculator_args(data[0])
    alerce_object = calculate_magnitude_statistics(*args)[0]
    print(alerce_object.magstats)
    assert len(alerce_object.magstats) > 0
