import math
import pandas as pd
from unittest import mock
from magstats_step.core.utils.compose import compose as super_magstats_calculator
from magstats_step.core.utils.magstats_intersection import CALCULATORS_LIST, magstats_intersection
from magstats_step.core.calculators import *
from data.messages import data
from data.utils import setup_blank_dto


def test_magstats_intersection():
    excluded_calcs = ["dmdt", "mjd", "ndet", "stellar"]
    result = magstats_intersection(excluded_calcs)
    expected_result = [calculate_ra, calculate_dec]
    assert set(result.values()) == set(expected_result)


def test_super_magstats_calculator():
    ra_calculator = mock.MagicMock()
    dec_calculator = mock.MagicMock()
    functions = [ra_calculator, dec_calculator]
    super_magstats_calculator(*functions)(data)
    ra_calculator.assert_called_with(dec_calculator(data))


def test_magstats_calculators_composition():
    excluded_calcs = list(set(CALCULATORS_LIST) - set(["ra", "dec"]))
    calculators = magstats_intersection(excluded_calcs)
    data_dto = setup_blank_dto(data[0])
    result_dto = super_magstats_calculator(*calculators.values())(data_dto)

    assert result_dto.alerce_object["meanra"] != -999
    assert result_dto.alerce_object["meandec"] != -999


def test_calculate_stats_coordinates():
    coords = pd.Series([250, 250, 250])
    e_coords = pd.Series([3600, 3600, 3600])
    expected_result = (250, math.sqrt(1 / 3) * 3600)
    result = calculate_stats_coordinates(coords, e_coords)
    assert result == expected_result


def test_calculate_dec():
    object_dto = setup_blank_dto(data[0])
    result_dto = calculate_dec(object_dto)
    assert result_dto.alerce_object["meandec"] != -999
    assert result_dto.alerce_object["sigmadec"] != -999


def test_calculate_ra():
    object_dto = setup_blank_dto(data[0])

    result_dto = calculate_ra(object_dto)
    assert result_dto.alerce_object["meanra"] != -999
    assert result_dto.alerce_object["sigmara"] != -999


def test_calculate_mjd():
    object_dto = setup_blank_dto(data[0])
    result_dto = calculate_mjd(object_dto)

    assert result_dto.alerce_object["firstmjd"] < result_dto.alerce_object["lastmjd"]


def test_calculate_ndet():
    object_dto = setup_blank_dto(data[0])
    result_dto = calculate_ndet(object_dto)

    assert result_dto.alerce_object["ndet"] != -999


def test_calculate_magnitude_statistics():
    object_dto = setup_blank_dto(data[0])
    result_dto = calculate_magnitude_statistics(object_dto)

    assert result_dto
