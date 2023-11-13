import os
import sys
import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import config

level = logging.INFO
if os.getenv("LOGGING_DEBUG"):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from lc_classification.core import LateClassifier

step_config = config()

prometheus_metrics = None
if step_config["PROMETHEUS"]:
    from prometheus_client import start_http_server
    from apf.metrics.prometheus import PrometheusMetrics

    prometheus_metrics = PrometheusMetrics()
    start_http_server(8000)

step = LateClassifier(
    config=step_config, level=level, prometheus_metrics=prometheus_metrics
)
step.start()
