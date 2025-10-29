import os
import sys
import logging
from features.database import PSQLConnection
from apf.core.settings import config_from_yaml_file


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
if os.getenv("CONFIG_FROM_YAML"):
    STEP_CONFIG = config_from_yaml_file(os.getenv("CONFIG_YAML_PATH"))
else:
    from settings import STEP_CONFIG


level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG"):
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

from features.step import FeatureStep

if STEP_CONFIG["FEATURE_FLAGS"]["USE_PROFILING"]:
    from pyroscope import configure

    configure(
        application_name="step.Feature",
        server_address=STEP_CONFIG["PYROSCOPE_SERVER"],
    )
db_sql = PSQLConnection(STEP_CONFIG["DB_CONFIG"])
step = FeatureStep(config=STEP_CONFIG, db_sql=db_sql)
step.start()
