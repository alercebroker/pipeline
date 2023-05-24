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
    if os.getenv('LOGGING_DEBUG'):
        level = logging.DEBUG

    logger = logging.getLogger("alerce")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)7s %(name)36s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)

    logger.addHandler(handler)

    return MagstatsStep(config=step_config)


if __name__ == "__main__":
    step_factory().start()
