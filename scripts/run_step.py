import os
import sys

import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)

from settings import CONSUMER_CONFIG, STEP_CONFIG
from correction import Correction
from apf.consumers import AVROFileConsumer

level = logging.INFO
if 'LOGGING_DEBUG' in locals():
    if LOGGING_DEBUG:
        level=logging.DEBUG

logging.basicConfig(level=level,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)
consumer = AVROFileConsumer(config=CONSUMER_CONFIG)

step = Correction(consumer,config=STEP_CONFIG,level=level)
step.start()