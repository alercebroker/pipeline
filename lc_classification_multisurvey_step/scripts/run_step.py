import logging
import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

from apf.core.settings import config_from_yaml_file  # noqa: E402


def set_logger(settings):
    level = logging.DEBUG if settings.get("LOGGING_DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return level


def step_creator():
    if os.getenv("CONFIG_FROM_YAML", False):
        # CONFIG_YAML_PATH lets a local run point at a checked-out yaml, the way
        # feature_step/scripts/run_step.py does. The default is the path the
        # chart mounts the ConfigMap at, so the container is unaffected.
        settings = config_from_yaml_file(
            os.getenv("CONFIG_YAML_PATH", "/config/config.yaml")
        )
    else:
        from settings import config

        settings = config()

    level = set_logger(settings)

    if settings.get("FEATURE_FLAGS", {}).get("PROMETHEUS"):
        from prometheus_client import start_http_server

        start_http_server(8000)

    from lc_classification_multisurvey_step.step import LateClassifierMultisurvey

    return LateClassifierMultisurvey(config=settings, level=level)


def step():
    step_creator().start()


if __name__ == "__main__":
    step()
