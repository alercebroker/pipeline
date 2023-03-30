import pandas as pd
from magstats_step.core.objstats import ObjectStatistics

from data.messages import data


def test_calculate_weighted_coordinates():
    coords = pd.Series([250, 250, 250])
    e_coords = pd.Series([1, 1, 1])
    result = ObjectStatistics.weighted_mean(coords, 1 / e_coords ** 2)
    assert result == 250


def test_calculate_coordinates():
    calculator = ObjectStatistics(data[0]["detections"])
    result = calculator.calculate_ra()
    assert "meanra" in result and "sigmara" in result


def test_calculate_mjd():
    calculator = ObjectStatistics(data[0]["detections"])
    result1 = calculator.calculate_firstmjd()
    result2 = calculator.calculate_lastmjd()

    assert "firstmjd" in result1 and "lastmjd" in result2
    assert (result1["firstmjd"] <= result2["lastmjd"]).all()


def test_calculate_ndet():
    calculator = ObjectStatistics(data[0]["detections"])
    result = calculator.calculate_ndet()

    sums = {}
    for det in data[0]["detections"]:
        sums[det["aid"]] = 1 if det["aid"] not in sums else sums[det["aid"]] + 1

    assert "ndet" in result
    assert (result["ndet"] == list(sums.values())).all()
