import unittest
import os
from apf.metrics import LogfileMetricsProducer


def test_no_path():
    config = {}
    log_metrics = LogfileMetricsProducer(config)


def test_path():
    test_path = "/tmp/test_log/log.log"
    if os.path.exists(test_path):
        os.remove(test_path)

    config = {"PATH": test_path}
    log_metrics = LogfileMetricsProducer(config)
    metrics = {"oid": "TEST", "candid": 1}
    log_metrics.send_metrics(metrics)

    metrics_list = {"oid": ["TEST1", "TEST2"], "candid": [1, 2]}
    log_metrics.send_metrics(metrics_list)
    assert os.path.exists(test_path)
    os.remove(test_path)
