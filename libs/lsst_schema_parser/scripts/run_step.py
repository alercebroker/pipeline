import logging
import os
import sys
import pathlib


PACKAGE_PATH = pathlib.Path(__file__).parent.parent
print("here\n", PACKAGE_PATH)
CONFIG_PATH = PACKAGE_PATH / "config.yaml"
print("here\n", CONFIG_PATH)
sys.path.append(PACKAGE_PATH)

from apf.core.settings import config_from_yaml_file

STEP_CONFIG = config_from_yaml_file(CONFIG_PATH)["STEP_CONFIG"]

level = logging.INFO
if STEP_CONFIG.get("LOGGING_DEBUG", False):
    level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from schema_parser.step import SchemaParserStep

step = SchemaParserStep(
    config=STEP_CONFIG
)
step.start()
