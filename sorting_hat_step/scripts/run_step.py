import os
import sys

import logging
import pyroscope

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics
from sorting_hat_step.database import MongoConnection, PsqlConnection


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import STEP_CONFIG

level = logging.INFO
if os.getenv("LOGGING_DEBUG"):
    level = logging.DEBUG

logger = logging.getLogger("alerce")
logger.setLevel(level)

fmt = logging.Formatter(
    "%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(fmt)
handler.setLevel(level)

logger.addHandler(handler)

from sorting_hat_step.step import SortingHatStep

mongo_database = MongoConnection(STEP_CONFIG["MONGO_CONFIG"])

if bool(os.getenv("USE_PROFILING", True)):
    logger.info("Configuring Pyroscope profiling...")
    pyroscope.configure(
        application_name="steps.SortingHat",
        server_address=os.getenv("PYROSCOPE_SERVER"),
    )

prometheus_metrics = PrometheusMetrics()
start_http_server(8000)

step = SortingHatStep(
    mongo_connection=mongo_database,
    config=STEP_CONFIG,
    prometheus_metrics=prometheus_metrics,
)
if STEP_CONFIG["USE_PSQL"].lower() == "true":
    psql_database = PsqlConnection(STEP_CONFIG["PSQL_CONFIG"])
    step.set_psql_driver(psql_database)
step.start()