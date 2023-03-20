from unittest import mock
from magstats_step.core.utils.compose import compose as super_magstats_calculator
from magstats_step.core.utils.magstats_intersection import magstats_intersection
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
