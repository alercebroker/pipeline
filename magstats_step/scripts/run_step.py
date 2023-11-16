import logging
import os
import sys

from apf.core.settings import config_from_yaml_file

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


def set_logger(config):
    level = logging.INFO
    if config.get("LOGGING_DEBUG"):
        level = logging.DEBUG

    logger = logging.getLogger("alerce")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)

    logger.addHandler(handler)
    return logger


def step_factory():
    from magstats_step.step import MagstatsStep

    if os.getenv("CONFIG_FROM_YAML"):
        step_config = config_from_yaml_file("/config/config.yaml")
    else:
        from settings import settings_factory

        step_config = settings_factory()

    set_logger(step_config)
    return MagstatsStep(config=step_config)


if __name__ == "__main__":
    step_factory().start()
