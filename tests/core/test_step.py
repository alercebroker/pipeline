from apf.core.step import GenericStep
import pytest

def test_get_single_extra_metrics():
    config = {
        "METRICS_CONFIG":{
            "CLASS": "apf.metrics.GenericMetricsProducer",
            "PARAMS": {},
            "EXTRA_METRICS": ["oid", "candid"]
        }
    }
    message = {"oid": "TEST", "candid": 1}
    gs = GenericStep(config=config)
    extra_metrics = gs.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is str
    assert type(extra_metrics["candid"]) is int
    assert type(extra_metrics["n_messages"]) is int


def test_get_batch_extra_metrics():
    config = {
        "METRICS_CONFIG":{
            "CLASS": "apf.metrics.GenericMetricsProducer",
            "PARAMS": {},
            "EXTRA_METRICS": ["oid", "candid", {"key": "candid", "alias": "str_candid", "format": lambda x: str(x)}]
        }
    }
    message = [{"oid": "TEST", "candid": 1}, {"oid": "TEST2"}, {"candid": 3}]
    gs = GenericStep(config=config)
    extra_metrics = gs.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is list
    assert type(extra_metrics["candid"]) is list
    assert type(extra_metrics["n_messages"]) is int


def test_get_value():
    config = {
        "METRICS_CONFIG":{
            "CLASS": "apf.metrics.GenericMetricsProducer",
            "PARAMS": {},
            "EXTRA_METRICS": ["oid", "candid"]
        }
    }
    message = {"oid": "TEST", "candid": 1}
    gs = GenericStep(config=config)

    aliased_metric, value = gs.get_value(message, "oid")
    assert( aliased_metric == "oid")
    assert(value == "TEST")

    aliased_metric, value = gs.get_value(message, "candid")
    assert( aliased_metric == "candid")
    assert(value == 1)

    aliased_metric, value = gs.get_value(message, {"key":"oid"})
    assert( aliased_metric == "oid")
    assert(value == "TEST")

    aliased_metric, value = gs.get_value(message, {"key":"oid", "alias": "new_oid"})
    assert( aliased_metric == "new_oid")
    assert(value == "TEST")

    aliased_metric, value = gs.get_value(message, {"key":"oid", "format": lambda x: x[0]})
    assert( aliased_metric == "oid")
    assert(value == "T")

    with pytest.raises(KeyError):
        gs.get_value(message, {})

    with pytest.raises(ValueError):
        gs.get_value(message, {"key": "oid", "format": "test"})

    with pytest.raises(ValueError):
        gs.get_value(message, {"key": "oid", "alias": 1})
