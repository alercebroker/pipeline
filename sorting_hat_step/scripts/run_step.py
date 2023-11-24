import os
import sys

import logging
import pyroscope

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics
from sorting_hat_step.database import MongoConnection, PsqlConnection
from apf.core.settings import config_from_yaml_file
from credentials import get_credentials


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

if os.getenv("CONFIG_FROM_YAML"):
    STEP_CONFIG = config_from_yaml_file("/config/config.yaml")
    STEP_CONFIG["METRICS_CONFIG"]["EXTRA_METRICS"] = [
        {"key": "candid", "format": lambda x: str(x)}
    ]
else:
    from settings import STEP_CONFIG

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG"):
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

if STEP_CONFIG.get("MONGO_SECRET_NAME"):
    mongo_database = MongoConnection(
        get_credentials(STEP_CONFIG["MONGO_SECRET_NAME"], secret_type="mongo")
    )
else:
    mongo_database = MongoConnection(STEP_CONFIG["MONGO_CONFIG"])

if bool(STEP_CONFIG["FEATURE_FLAGS"].get("USE_PROFILING", True)):
    logger.info("Configuring Pyroscope profiling...")
    pyroscope.configure(
        application_name="steps.SortingHat",
        server_address=STEP_CONFIG.get("PYROSCOPE_SERVER", ""),
    )

prometheus_metrics = None
prometheus_metrics = PrometheusMetrics()
if STEP_CONFIG["FEATURE_FLAGS"]["PROMETHEUS"]:
    start_http_server(8000)

step = SortingHatStep(
    mongo_connection=mongo_database,
    config=STEP_CONFIG,
    prometheus_metrics=prometheus_metrics,
)
if STEP_CONFIG["FEATURE_FLAGS"]["USE_PSQL"]:
    if STEP_CONFIG.get("PSQL_SECRET_NAME"):
        psql_database = PsqlConnection(
            get_credentials(STEP_CONFIG["PSQL_SECRET_NAME"], secret_type="psql")
        )
    else:
        psql_database = PsqlConnection(STEP_CONFIG["PSQL_CONFIG"])
    step.set_psql_driver(psql_database)
step.start()
