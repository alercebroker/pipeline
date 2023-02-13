import os
import sys

import logging

from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator

from apf.producers.kafka import KafkaProducer

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


from sorting_hat_step import SortingHatStep
from apf.core import get_class

if "CLASS" in CONSUMER_CONFIG:
    Consumer = get_class(CONSUMER_CONFIG["CLASS"])
else:
    from apf.consumers import KafkaConsumer as Consumer

consumer = Consumer(config=CONSUMER_CONFIG)
database = new_DBConnection(MongoDatabaseCreator)

if "CLASS" in PRODUCER_CONFIG:
    producer_class = get_class(PRODUCER_CONFIG["CLASS"])
    producer = producer_class(PRODUCER_CONFIG)
else:
    producer = KafkaProducer(PRODUCER_CONFIG)

step = SortingHatStep(consumer, STEP_CONFIG, producer, database, level=level)
step.start()
