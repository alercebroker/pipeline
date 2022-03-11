import os
import sys

import logging
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)
from settings import *

level = logging.INFO
if os.getenv('LOGGING_DEBUG'):
    level = logging.DEBUG

logging.basicConfig(level=level,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)


from xmatch_step import XmatchStep
from apf.consumers import KafkaConsumer as Consumer

consumer = Consumer(config=CONSUMER_CONFIG)

step = XmatchStep(consumer, config=STEP_CONFIG, level=level)
step.start()
