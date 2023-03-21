import math
from unittest import mock
from magstats_step.core.utils.compose import compose as super_magstats_calculator
from magstats_step.core.utils.magstats_intersection import magstats_intersection
from magstats_step.core.factories.object import alerce_object_factory
from magstats_step.core.utils.object_dto import ObjectDTO
from magstats_step.core.calculators import *
from data.messages import data


def test_magstats_intersection():
    magstat_list = ["dmdt", "ra", "dec"]
    result = magstats_intersection(magstat_list)
    expected_result = [calculate_dmdt, calculate_ra, calculate_dec]
    assert result== expected_result


def test_super_magstats_calculator():
    ra_calculator = mock.MagicMock()
    dec_calculator = mock.MagicMock()
    functions = [ra_calculator, dec_calculator]
    super_magstats_calculator(*functions)(data)
    ra_calculator.assert_called_with(dec_calculator(data))

def test_calculate_stats_coordinates():
    coords = [250, 250, 250]
    e_coords = [3600, 3600, 3600]
    expected_result = (250, math.sqrt(1/3) * 3600)
    result = calculate_stats_coordinates(coords, e_coords)
    assert result == expected_result

def test_calculate_dec():
    detections, non_detections = data[0]["detections"], data[0]["non_detections"]
    alerce_object = alerce_object_factory(data[0])
    object_dto = ObjectDTO(alerce_object, detections, non_detections)

    result_dto = calculate_dec(object_dto)
    assert result_dto.alerce_object["meandec"] != -999
    assert result_dto.alerce_object["sigmadec"] != -999

def test_calculate_ra():
    detections, non_detections = data[0]["detections"], data[0]["non_detections"]
    alerce_object = alerce_object_factory(data[0])
    object_dto = ObjectDTO(alerce_object, detections, non_detections)

    result_dto = calculate_ra(object_dto)
    assert result_dto.alerce_object["meanra"] != -999
    assert result_dto.alerce_object["sigmara"] != -999