import os
import sys
import logging
from apf.core.settings import config_from_yaml_file
from apf.metrics.prometheus import DefaultPrometheusMetrics

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import config

level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from ms_classification_step.core.step import MultiStampClassifier

step_config = config_from_yaml_file("/home/bastiancito/Desktop/config.yaml")


if step_config["FEATURE_FLAGS"]["USE_PROFILING"]:
    from pyroscope import configure

    configure(
        application_name=f"step.LCClassification{step_config['MODEL_CONFIG']['NAME']}",
        server_address=step_config["PYROSCOPE_SERVER"],
    )

prometheus_metrics = DefaultPrometheusMetrics()
if step_config["FEATURE_FLAGS"]["PROMETHEUS"]:
    from prometheus_client import start_http_server
    from apf.metrics.prometheus import PrometheusMetrics

    prometheus_metrics = PrometheusMetrics()
    start_http_server(8000)

step = MultiStampClassifier(
    config=step_config, level=level, prometheus_metrics=prometheus_metrics
)
step.start()
