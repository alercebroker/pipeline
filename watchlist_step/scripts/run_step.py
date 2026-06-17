import logging
import os
import sys

from apf.core.settings import config_from_yaml_file

from watchlist_step.step import WatchlistStep

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)


STEP_CONFIG = config_from_yaml_file(
    os.getenv("CONFIG_YAML_PATH", "/config/config.yaml")
)

# EXTRA_METRICS carry callables, which can't be expressed in the YAML config.
if "METRICS_CONFIG" in STEP_CONFIG:
    STEP_CONFIG["METRICS_CONFIG"]["EXTRA_METRICS"] = [
        {"key": "candid", "format": lambda x: str(x)},
        {"key": "objectId", "alias": "oid"},
    ]


# Secrets are sourced from the environment (k8s Secrets via secretKeyRef), not
# baked into the mounted config.yaml. Overlay each present env var onto the
# config; absent vars leave the yaml value untouched.
def _overlay_secret(path, env_name):
    value = os.getenv(env_name)
    if value is None:
        return
    node = STEP_CONFIG
    for key in path[:-1]:
        if key not in node:
            return
        node = node[key]
    node[path[-1]] = value


_overlay_secret(["CONSUMER_CONFIG", "PARAMS", "sasl.username"], "CONSUMER_KAFKA_USERNAME")
_overlay_secret(["CONSUMER_CONFIG", "PARAMS", "sasl.password"], "CONSUMER_KAFKA_PASSWORD")
_overlay_secret(["METRICS_CONFIG", "PARAMS", "PARAMS", "sasl.username"], "METRICS_KAFKA_USERNAME")
_overlay_secret(["METRICS_CONFIG", "PARAMS", "PARAMS", "sasl.password"], "METRICS_KAFKA_PASSWORD")
_overlay_secret(["PSQL_CONFIG", "USER"], "USERS_DB_USER")
_overlay_secret(["PSQL_CONFIG", "PASSWORD"], "USERS_DB_PASSWORD")

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG"):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if bool(STEP_CONFIG.get("FEATURE_FLAGS", {}).get("PROMETHEUS", False)):
    from prometheus_client import start_http_server

    start_http_server(8000)

step = WatchlistStep(
    config=STEP_CONFIG,
    strategy_name=STEP_CONFIG["UPDATE_STRATEGY"],
    level=level,
)
step.start()
