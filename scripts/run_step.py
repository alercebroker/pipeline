import os
import sys

import logging

from apf.producers import KafkaProducer

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)
from settings import (
    CONSUMER_CONFIG,
    OUTPUT_PRODUCER_CONFIG,
    SCRIBE_PRODUCER_CONFIG,
    CLASSIFIER_STRATEGY,
)
from stamp_classifier_step.strategies import get_strategy

level = logging.INFO
if "LOGGING_DEBUG" in locals():
    if locals()["LOGGING_DEBUG"]:
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

consumer = Consumer(config=CONSUMER_CONFIG)
output_producer = KafkaProducer(config=OUTPUT_PRODUCER_CONFIG)
scribe_producer = KafkaProducer(config=SCRIBE_PRODUCER_CONFIG)
strategy = get_strategy(CLASSIFIER_STRATEGY)

step = StampClassifierStep(
    consumer,
    producer=output_producer,
    scribe_producer=scribe_producer,
    strategy=strategy,
    level=level,
)
step.start()
