from unittest.mock import MagicMock
from data.messages import data
from data.utils import setup_blank_dto
from scripts.run_step import step_factory
from pytest import fixture


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
