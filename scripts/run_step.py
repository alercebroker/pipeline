import os
import sys
import logging

from prometheus_client import start_http_server
from apf.metrics.prometheus import PrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


from prv_candidates_step import PrvCandidatesStep


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

    prometheus_metrics = PrometheusMetrics()
    start_http_server(8000)
    return PrvCandidatesStep(config=settings, level=level, prometheus_metrics=prometheus_metrics)


if __name__ == "__main__":
    step_creator().start()
