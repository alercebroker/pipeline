import logging
from apf.metrics.prometheus import PrometheusMetrics, DefaultPrometheusMetrics
from prometheus_client import start_http_server

try:
    from correction import CorrectionStep
except ImportError:
    import os
    import sys

    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
    sys.path.append(PACKAGE_PATH)
    from correction import CorrectionStep


def step_creator():
    from settings import settings_creator

    settings = settings_creator()
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if settings["PROMETHEUS"]:
        start_http_server(8000)
        prometheus_metrics = PrometheusMetrics()
    else:
        prometheus_metrics = DefaultPrometheusMetrics()
    if settings["LOGGING_DEBUG"]:
        level = logging.DEBUG

    return CorrectionStep(
        config=settings, level=level, prometheus_metrics=prometheus_metrics
    )


if __name__ == "__main__":
    step_creator().start()
