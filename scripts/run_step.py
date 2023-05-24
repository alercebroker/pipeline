import os
import sys
import logging

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import *

level = logging.INFO
if os.getenv('LOGGING_DEBUG'):
    level = logging.DEBUG

logger = logging.getLogger("alerce")
logger.setLevel(level)

fmt = logging.Formatter("%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
handler.setLevel(level)

logger.addHandler(handler)

from xmatch_step import XmatchStep

prometheus_metrics = PrometheusMetrics()
start_http_server(8000)

step = XmatchStep(
    config=STEP_CONFIG, prometheus_metrics=prometheus_metrics
)
step.start()
