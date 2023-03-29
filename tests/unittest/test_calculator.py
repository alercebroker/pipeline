import pandas as pd
from magstats_step.core.objstats import ObjectStatistics

from data.messages import data


def test_calculate_weighted_coordinates():
    coords = pd.Series([250, 250, 250])
    e_coords = pd.Series([1, 1, 1])
    result = ObjectStatistics.weighted_mean(coords, 1 / e_coords ** 2)
    assert result == 250


def test_calculate_coordinates():
    calculator = ObjectStatistics(**data[0])
    result = calculator.calculate_coordinates()
    assert "meandec" in result and "sigmadec" in result
    assert "meanra" in result and "sigmara" in result


def test_calculate_mjd():
    calculator = ObjectStatistics(**data[0])
    result = calculator.calculate_mjd()

    assert "firstmjd" in result and "lastmjd" in result
    assert result["firstmjd"] < result["lastmjd"]


def test_calculate_ndet():
    calculator = ObjectStatistics(**data[0])
    result = calculator.calculate_ndet()

    assert "ndet" in result
    assert result["ndet"] == len(data[0]["detections"])
