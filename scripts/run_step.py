import os
import sys

import logging
import pyroscope

from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
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

from sorting_hat_step import SortingHatStep

database = new_DBConnection(MongoDatabaseCreator)

if bool(os.getenv("USE_PROFILING", True)):
    pyroscope.configure(application_name="steps.SortingHat", server_address=os.getenv("PYROSCOPE_SERVER"))

prometheus_metrics = PrometheusMetrics()
start_http_server(8000)

step = SortingHatStep(
    db_connection=database,
    config=STEP_CONFIG,
    prometheus_metrics=prometheus_metrics,
)
step.start()
