import logging
import os
import sys

import pyroscope
from apf.core.settings import config_from_yaml_file
from apf.metrics.prometheus import PrometheusMetrics
from prometheus_client import start_http_server

from ingestion_step.step import SortingHatStep

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


STEP_CONFIG = config_from_yaml_file("config.yaml")
STEP_CONFIG["METRICS_CONFIG"]["EXTRA_METRICS"] = [{"key": "candid", "format": str}]

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


if bool(STEP_CONFIG["FEATURE_FLAGS"].get("USE_PROFILING", True)):
    logger.info("Configuring Pyroscope profiling...")
    pyroscope.configure(  # pyright: ignore
        application_name="steps.SortingHat",
        server_address=STEP_CONFIG.get("PYROSCOPE_SERVER", ""),
    )

prometheus_metrics = None
prometheus_metrics = PrometheusMetrics()
if STEP_CONFIG["FEATURE_FLAGS"]["PROMETHEUS"]:
    start_http_server(8000)

step = SortingHatStep(
    config=STEP_CONFIG,
    prometheus_metrics=prometheus_metrics,
)

step.start()
