import os
import sys
import logging
from apf.core.settings import config_from_yaml_file
from s3_step.step import S3Step
from apf.metrics.prometheus import PrometheusMetrics
from prometheus_client import start_http_server


def set_logger(settings):
    level = logging.INFO
    if settings.get("LOGGING_DEBUG"):
        level = logging.DEBUG

    logger = logging.getLogger("alerce")
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)7s %(name)36s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    handler.setLevel(level)

    logger.addHandler(handler)
    return logger


def step_creator():
    settings = config_from_yaml_file("/config/config.yaml")

    logger = set_logger(settings)

    step_params = {"config": settings}

    if settings["FEATURE_FLAGS"]["PROMETHEUS"]:
        step_params["prometheus_metrics"] = PrometheusMetrics()
        start_http_server(8000)

    if settings["FEATURE_FLAGS"]["USE_PROFILING"]:
        from pyroscope import configure

        configure(
            application_name="step.LightcurveCorrection",
            server_address=settings["PYROSCOPE_SERVER"],
        )

    return S3Step(**step_params)


if __name__ == "__main__":
    step_creator().start()
