import logging
import os
import sys

from apf.core.settings import config_from_yaml_file
from prometheus_client import start_http_server

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

from hips_dummy_step.step import HipsDummyStep  # noqa: E402


def step():
    config = config_from_yaml_file(
        os.getenv("CONFIG_YAML_PATH", "/config/config.yaml")
    )

    level = logging.DEBUG if config.get("LOGGING_DEBUG") else logging.INFO
    logger = logging.getLogger("alerce")
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)7s %(name)36s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(level)
    logger.addHandler(handler)

    if config.get("FEATURE_FLAGS", {}).get("PROMETHEUS", False):
        start_http_server(8000)

    HipsDummyStep(config=config).start()


if __name__ == "__main__":
    step()
