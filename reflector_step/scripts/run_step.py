import os
import sys

import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import *

level = logging.INFO
if "LOGGING_DEBUG" in locals():
    if LOGGING_DEBUG:
        level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from reflector_step.step import CustomMirrormaker

keep_original_timestamp = STEP_CONFIG.pop("keep_original_timestamp")
use_message_topic = STEP_CONFIG.pop("use_message_topic")

step = CustomMirrormaker(
    config=STEP_CONFIG,
    keep_original_timestamp=keep_original_timestamp,
    use_message_topic=use_message_topic,
)

step.start()
