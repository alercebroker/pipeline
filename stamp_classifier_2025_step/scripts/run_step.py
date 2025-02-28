import os
import sys
import logging
from apf.core.settings import config_from_yaml_file
from apf.metrics.prometheus import DefaultPrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

level = logging.INFO

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from stamp_classifier_2025_step.core.step import MultiScaleStampClassifier


step_config = config_from_yaml_file(os.path.join(SCRIPT_PATH, "config.yaml"))
prometheus_metrics = DefaultPrometheusMetrics()

if step_config["FEATURE_FLAGS"]["PROMETHEUS"]:
    from prometheus_client import start_http_server
    from apf.metrics.prometheus import PrometheusMetrics

    prometheus_metrics = PrometheusMetrics()
    start_http_server(8000)

step = MultiScaleStampClassifier(
    config=step_config, level=level, prometheus_metrics=prometheus_metrics
)
step.start()
