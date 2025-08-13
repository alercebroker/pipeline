import os
import sys
import logging
from apf.core.settings import config_from_yaml_file
from apf.metrics.prometheus import DefaultPrometheusMetrics

from rubin_stamp_classifier_step.step import StampClassifierStep


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

step_config = config_from_yaml_file(os.getenv("CONFIG_YAML_PATH"))

logging.basicConfig(
    level=step_config["LOGGING_LEVEL"],
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

prometheus_metrics = DefaultPrometheusMetrics()

if step_config["FEATURE_FLAGS"]["PROMETHEUS"]:
    from prometheus_client import start_http_server
    from apf.metrics.prometheus import PrometheusMetrics

    prometheus_metrics = PrometheusMetrics()
    start_http_server(8000)

step = StampClassifierStep(
    config=step_config,
    level=step_config["LOGGING_LEVEL"],
    prometheus_metrics=prometheus_metrics,
)
step.start()
