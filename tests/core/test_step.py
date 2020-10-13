from apf.core.step import GenericStep


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
            "EXTRA_METRICS": ["oid", "candid"]
        }
    }
    message = [{"oid": "TEST", "candid": 1}, {"oid": "TEST2"}, {"candid": 3}]
    gs = GenericStep(config=config)
    extra_metrics = gs.get_extra_metrics(message)
    assert type(extra_metrics) is dict
    assert type(extra_metrics["oid"]) is list
    assert type(extra_metrics["candid"]) is list
    assert type(extra_metrics["n_messages"]) is int
