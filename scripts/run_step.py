import os
import sys

import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


def step_factory():
    from magstats_step.step import MagstatsStep
    from settings import settings_factory

    step_config = settings_factory()

    level = logging.INFO
    if step_config["LOGGING_DEBUG"]:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return MagstatsStep(config=step_config, level=level)


if __name__ == "__main__":
    step_factory().start()
