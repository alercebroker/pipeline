import logging
import os
import sys
from credentials import get_credentials

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

if os.getenv("CONFIG_FROM_YAML"):
    from apf.core.settings import config_from_yaml_file

    STEP_CONFIG = config_from_yaml_file("/config/config.yaml")
else:
    sys.path.append(PACKAGE_PATH)
    from settings import STEP_CONFIG

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG"):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from metadata_step import MetadataStep
from metadata_step.utils.database import PSQLConnection


# DB credentials. Cloud (AWS legacy): a Secrets Manager secret name resolves the
# full connection. On-prem (quimal): no Secrets Manager access, so the non-sensitive
# connection fields come from the mounted config.yaml (STEP_CONFIG["PSQL_CONFIG"]) and
# only the password is injected from a Kubernetes Secret via the PSQL_PASSWORD env var.
if STEP_CONFIG.get("DATABASE_SECRET_NAME"):
    DATABASE = get_credentials(STEP_CONFIG["DATABASE_SECRET_NAME"])
else:
    DATABASE = dict(STEP_CONFIG["PSQL_CONFIG"])
    DATABASE["PASSWORD"] = os.environ["PSQL_PASSWORD"]
sql = PSQLConnection(DATABASE, echo=STEP_CONFIG.get("LOGGING_DEBUG", False))

step = MetadataStep(config=STEP_CONFIG, db_sql=sql)
step.start()
