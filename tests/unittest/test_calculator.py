from .data.messages import data
from unittest import mock


def test_magstats_intersection():
    magstat_list = ["dmdt", "ra", "dec"]
    result = magstats_intersection(magstat_list)
    expected_result = [dmdt_calculator, ra_calculator, dec_calculator]
    assert result == expected_result


def test_super_magstats_calculator():
    ra_calculator = mock.MagicMock()
    dec_calculator = mock.MagicMock()
    functions = [ra_calculator, dec_calculator]
    super_magtats_calculator(functions)(data)
    ra_calculator.assert_called_with(dec_calculator(data))
