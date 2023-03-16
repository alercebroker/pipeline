from unittest.mock import MagicMock
from data.messages import data
from scripts.run_step import step_factory


def test_execute():
    step = step_factory()
    result = step.execute(data)
    for i, d in enumerate(data):
        step.magstats_calculator.assert_called_with(d)
        assert d["aid"] == result[i]["aid"]
        assert isinstance(result[i]["meanra"], MagicMock)
        assert isinstance(result[i]["meandec"], MagicMock)
        assert isinstance(result[i]["magstats"], MagicMock)
        assert isinstance(result[i]["oid"], MagicMock)
        assert isinstance(result[i]["tid"], MagicMock)
        assert isinstance(result[i]["firstmjd"], MagicMock)
        assert isinstance(result[i]["lastmjd"], MagicMock)
        assert isinstance(result[i]["ndet"], MagicMock)
        assert isinstance(result[i]["sigmara"], MagicMock)
        assert isinstance(result[i]["sigmadec"], MagicMock)


def test_object_creator():
    step = step_factory()
    object = step.object_creator(data[0])
    assert object["aid"] == data[0]["aid"]
    assert object["meanra"] == -999
    assert object["meandec"] == -999
    assert object["magstats"] == []
    assert object["oid"] == []
    assert object["tid"] == []
    assert object["firstmjd"] == -999
    assert object["lastmjd"] == -999
    assert object["ndet"] == -999
    assert object["sigmara"] == -999
    assert object["sigmadec"] == -999
