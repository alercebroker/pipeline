import logging
import os

from apf.core.settings import config_from_yaml_file

from alerts_store.step import AlertStore

STEP_CONFIG = config_from_yaml_file(
    os.getenv("CONFIG_YAML_PATH", "/config/config.yaml")
)

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


step = AlertStore(
    config=STEP_CONFIG,
)

step.start()
