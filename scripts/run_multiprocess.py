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

from xmatch_step import XmatchStep
from apf.consumers import GenericConsumer as Consumer

n_process = STEP_CONFIG.get("N_PROCESS", 1)


def create_and_run(idx, Consumer):
    CONSUMER_CONFIG["ID"] = idx
    consumer = Consumer(config=CONSUMER_CONFIG)
    step = XmatchStep(consumer, config=STEP_CONFIG, level=level)
    step.start()


process_list = []
for i in range(n_process):
    process_list.append(Process(target=create_and_run, args=(i, Consumer)))

[p.start() for p in process_list]
[p.join() for p in process_list]
