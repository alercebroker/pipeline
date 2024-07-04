import logging
import os
from apf.metrics.prometheus import PrometheusMetrics
from apf.core.settings import config_from_yaml_file
from credentials import get_credentials
from prometheus_client import start_http_server

from lightcurve_step.step import LightcurveStep
from lightcurve_step.database_mongo import DatabaseConnection
from lightcurve_step.database_sql import PSQLConnection


def set_logger(settings):
    level = logging.INFO
    if settings.get("LOGGING_DEBUG"):
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
    return logger


def step_creator():
    if os.getenv("CONFIG_FROM_YAML", False):
        settings = config_from_yaml_file("/config/config.yaml")
    else:
        from settings import settings_creator

        settings = settings_creator()

    logger = set_logger(settings)

    db_mongo = None
    if settings["FEATURE_FLAGS"]["USE_MONGO"]:
        if settings.get("MONGO_SECRET_NAME"):
            db_mongo = DatabaseConnection(
                get_credentials(settings["MONGO_SECRET_NAME"], "mongo")
            )
        else:
            db_mongo = DatabaseConnection(settings["MONGO_CONFIG"])

    db_sql = None
    if settings["FEATURE_FLAGS"]["USE_SQL"]:
        if settings.get("SQL_SECRET_NAME"):
            db_sql = PSQLConnection(
                get_credentials(settings["SQL_SECRET_NAME"], "sql"),
                echo=settings["LOGGING_DEBUG"],
            )
        else:
            db_sql = PSQLConnection(settings["PSQL_CONFIG"])

    step_params = {"config": settings, "db_mongo": db_mongo, "db_sql": db_sql}

    if settings["FEATURE_FLAGS"]["PROMETHEUS"]:
        step_params["prometheus_metrics"] = PrometheusMetrics()
        start_http_server(8000)

    if settings["FEATURE_FLAGS"]["USE_PROFILING"]:
        from pyroscope import configure

        configure(
            application_name="step.Lightcurve",
            server_address=settings["PYROSCOPE_SERVER"],
        )

    if settings["FEATURE_FLAGS"]["SKIP_MJD_FILTER"]:
        logger.info(
            "This step won't filter detections by MJD. \
            Keep this in mind when using for ELAsTiCC"
        )

    return LightcurveStep(**step_params)


if __name__ == "__main__":
    step_creator().start()
