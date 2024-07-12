import logging
import os
import sys

from credentials import get_credentials

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

if os.getenv("CONFIG_FROM_YAML"):
    from apf.core.settings import config_from_yaml_file

    STEP_CONFIG = config_from_yaml_file("/config/config.yaml")

else:
    from settings import STEP_CONFIG

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG", False):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from apf.metrics.prometheus import PrometheusMetrics
from mongo_scribe import MongoScribe
from prometheus_client import start_http_server

# Pyroscope config
use_profiling = STEP_CONFIG.pop("USE_PROFILING")
pyroscope_server = STEP_CONFIG.pop("PYROSCOPE_SERVER")
# db config
db_type = STEP_CONFIG.pop("DB_TYPE")
DB_CONFIG = {}
if STEP_CONFIG.get("DB_SECRET_NAME", None):
    db_credentials = get_credentials(STEP_CONFIG["DB_SECRET_NAME"], db_type)
    if db_type == "mongo":
        DB_CONFIG["MONGO"] = db_credentials
    elif db_type == "sql":
        DB_CONFIG["PSQL"] = db_credentials
else:
    DB_CONFIG=STEP_CONFIG.get("DB_CONFIG")
STEP_CONFIG["DB_CONFIG"] = DB_CONFIG

if use_profiling:
    from pyroscope import configure

    configure(
        application_name="step.ScribeStep", server_address=pyroscope_server
    )
# PROMETHEUS
prometheus_metrics = PrometheusMetrics()
if STEP_CONFIG.get("FEATURE_FLAGS", {}).get("PROMETHEUS"):
    start_http_server(8000)

step = MongoScribe(
    config=STEP_CONFIG, db=db_type, prometheus_metrics=prometheus_metrics
)
step.start()
