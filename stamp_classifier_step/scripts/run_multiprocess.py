import os
import sys

import logging
from multiprocessing import Process

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

from stamp_classifier_step import StampClassifierStep
from apf.core import get_class

if "CLASS" in CONSUMER_CONFIG:
    Consumer = get_class(CONSUMER_CONFIG["CLASS"])
else:
    from apf.consumers import KafkaConsumer as Consumer

n_process = STEP_CONFIG.get("N_PROCESS", 1)


def create_and_run(idx, Consumer):
    CONSUMER_CONFIG["ID"] = idx
    consumer = Consumer(config=CONSUMER_CONFIG)
    step = StampClassifierStep(consumer, config=STEP_CONFIG, level=level)
    step.start()


process_list = []
for i in range(n_process):
    process_list.append(Process(target=create_and_run, args=(i, Consumer)))

[p.start() for p in process_list]
[p.join() for p in process_list]
