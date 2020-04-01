import os
import sys
import logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH,".."))

sys.path.append(PACKAGE_PATH)

from settings import CONSUMER_CONFIG, STEP_CONFIG
from correction import Correction
from apf.consumers import KafkaConsumer

consumer = KafkaConsumer(config=CONSUMER_CONFIG)

step = Correction(consumer,config=STEP_CONFIG,level=level)
step.start()
