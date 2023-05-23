import os
import sys

import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import STEP_CONFIG

level = logging.INFO
if os.getenv("LOGGING_DEBUG"):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from mongo_scribe import MongoScribe
from apf.metrics.prometheus import PrometheusMetrics
from prometheus_client import start_http_server

prometheus_metrics = PrometheusMetrics()
start_http_server(8000)

step = MongoScribe(config=STEP_CONFIG, level=level, prometheus_metrics=prometheus_metrics)
step.start()
