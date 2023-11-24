import os
import sys
import logging

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics, DefaultPrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

if os.getenv("CONFIG_FROM_YAML"):
    from apf.core.settings import config_from_yaml_file

    STEP_CONFIG = config_from_yaml_file("/config/config.yaml")
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

from xmatch_step import XmatchStep

prometheus_metrics = DefaultPrometheusMetrics()
if STEP_CONFIG.get("FEATURE_FLAGS", {}).get("PROMETHEUS"):
    start_http_server(8000)
    prometheus_metrics = PrometheusMetrics()

if STEP_CONFIG.get("FEATURE_FLAGS", {}).get("USE_PROFILING", False):
    from pyroscope import configure

    configure(
        application_name="step.Lightcurve",
        server_address=STEP_CONFIG["PYROSCOPE_SERVER"],
    )

step = XmatchStep(config=STEP_CONFIG, prometheus_metrics=prometheus_metrics)
step.start()
