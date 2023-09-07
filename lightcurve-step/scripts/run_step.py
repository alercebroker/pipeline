import logging
import os
from apf.metrics.prometheus import PrometheusMetrics
from prometheus_client import start_http_server

from lightcurve_step.step import LightcurveStep
from lightcurve_step.database import DatabaseConnection


def step_creator():
    from settings import settings_creator

    settings = settings_creator()

    level = logging.INFO
    if os.getenv("LOGGING_DEBUG"):
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
    db = DatabaseConnection(settings["DB_CONFIG"])

    step_params = {
        "config": settings,
        "db_client": db,
        "prefix": os.getenv("CLASS_PREFIX", "")
    }

    if settings["PROMETHEUS"]:
        step_params["prometheus_metrics"] = PrometheusMetrics()
        start_http_server(8000)

    if settings["USE_PROFILING"]:
        from pyroscope import configure

        configure(
            application_name="step.Lightcurve",
            server_address=settings["PYROSCOPE_SERVER"],
        )

    return LightcurveStep(**step_params)


if __name__ == "__main__":
    step_creator().start()
