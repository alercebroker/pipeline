import os
import sys

import logging

level = logging.INFO
if 'LOGGING_DEBUG' in locals():
    if LOGGING_DEBUG:
        level=logging.DEBUG

logging.basicConfig(level=level,
                    format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)

from settings import *
from {{package_name}} import {{class_name}}
from apf.consumers import GenericConsumer
from apf.producers import GenericProducer

consumer = GenericConsumer(config=CONSUMER_CONFIG)
producer = GenericProducer(config=PRODUCER_CONFIG)

step = {{class_name}}(consumer,producer,config=STEP_CONFIG,level=level)
step.start()
