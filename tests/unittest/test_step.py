from magstats_step.core.factories.object import AlerceObject
from data.messages import data
from scripts.run_step import step_factory


def test_object_creator():
    step = step_factory()
    alerce_object = step.object_creator(data[0])
    assert isinstance(alerce_object, AlerceObject)
    assert alerce_object.aid == data[0]["aid"]
    assert alerce_object.meanra is None
    assert alerce_object.meandec is None
    assert alerce_object.magstats == []
    assert alerce_object.oid == []
    assert alerce_object.tid== []
    assert alerce_object.firstmjd is None
    assert alerce_object.lastmjd is None
    assert alerce_object.ndet is None
    assert alerce_object.sigmara is None
    assert alerce_object.sigmadec is None
    assert alerce_object.stellar is False
