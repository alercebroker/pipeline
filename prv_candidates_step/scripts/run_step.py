import logging
import os
import sys

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

from prv_candidates_step.step import PrvCandidatesStep

prometheus_metrics = PrometheusMetrics()


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

    if settings["USE_PROMETHEUS"]:
        start_http_server(8000)

    return PrvCandidatesStep(
        config=settings,
        prometheus_metrics=prometheus_metrics,
    )


if __name__ == "__main__":
    step_creator().start()
