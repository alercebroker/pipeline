import logging
import os
import sys

from apf.core.settings import config_from_yaml_file
from prometheus_client import start_http_server

from ingestion_step.step import IngestionStep

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


STEP_CONFIG = config_from_yaml_file(
    os.getenv("CONFIG_YAML_PATH", "/config/config.yaml")
)
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
    raise NotImplemented

if bool(STEP_CONFIG.get("FEATURE_FLAGS", {}).get("PROMETHEUS", False)):
    start_http_server(8000)

step = IngestionStep(config=STEP_CONFIG)

step.start()
