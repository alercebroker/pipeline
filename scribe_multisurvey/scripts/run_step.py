import logging
import os
import sys

from credentials import get_credentials

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

from apf.core.settings import config_from_yaml_file

STEP_CONFIG = config_from_yaml_file("/home/alex/Work/Projects/pipeline/scribe_multisurvey/scripts/config.yaml")

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG", False):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from apf.metrics.prometheus import PrometheusMetrics
from sql_scribe.step import SqlScribe
from prometheus_client import start_http_server

# Pyroscope config
use_profiling = STEP_CONFIG.pop("USE_PROFILING")
pyroscope_server = STEP_CONFIG.pop("PYROSCOPE_SERVER")

if use_profiling:
    from pyroscope import configure

    configure(
        application_name="step.ScribeStep", server_address=pyroscope_server
    )
# PROMETHEUS
prometheus_metrics = PrometheusMetrics()
if STEP_CONFIG.get("FEATURE_FLAGS", {}).get("PROMETHEUS"):
    start_http_server(8000)

step = SqlScribe(
    config=STEP_CONFIG, prometheus_metrics=prometheus_metrics
)
step.start()
