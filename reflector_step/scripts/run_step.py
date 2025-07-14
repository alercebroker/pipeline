import os
import sys

import logging
from apf.core.settings import config_from_yaml_file
from reflector_step.step import CustomMirrormaker


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

settings = config_from_yaml_file(os.getenv("CONFIG_YAML_PATH"))

logging.basicConfig(
    level=settings["LOGGING_LEVEL"],
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

step_config = settings["STEP_CONFIG"]

keep_original_timestamp = step_config.pop("keep_original_timestamp")
use_message_topic = step_config.pop("use_message_topic")

step = CustomMirrormaker(
    config=step_config,
    keep_original_timestamp=keep_original_timestamp,
    use_message_topic=use_message_topic,
)

step.start()
