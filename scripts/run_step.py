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

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if settings["LOGGING_DEBUG"]:
        level = logging.DEBUG
    if settings["PROMETHEUS"]:
        start_http_server(8000)

    db = MongoConnection()
    return LightcurveStep(config=settings, db_client=db)


if __name__ == "__main__":
    step_creator().start()
