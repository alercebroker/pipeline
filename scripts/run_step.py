import logging

from prometheus_client import start_http_server

from db_plugins.db.mongo.connection import MongoConnection

try:
    from lightcurve_step import LightcurveStep
except ImportError:
    import os
    import sys

    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
    sys.path.append(PACKAGE_PATH)
    from lightcurve_step import LightcurveStep


def step_creator():
    from settings import settings_creator

    settings = settings_creator()

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

    if settings["PROMETHEUS"]:
        start_http_server(8000)

    db = MongoConnection()
    return LightcurveStep(config=settings, db_client=db)


if __name__ == "__main__":
    step_creator().start()
