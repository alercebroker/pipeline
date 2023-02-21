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
    STEP_CONFIG,
)
from atlas_stamp_classifier_step.strategies.atlas import AtlasStrategy

level = logging.INFO
if "LOGGING_DEBUG" in locals():
    if locals()["LOGGING_DEBUG"]:
        level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


from atlas_stamp_classifier_step import AtlasStampClassifierStep
from apf.core import get_class

if "CLASS" in CONSUMER_CONFIG:
    Consumer = get_class(CONSUMER_CONFIG["CLASS"])
else:
    from apf.consumers import KafkaConsumer as Consumer

consumer = Consumer(config=CONSUMER_CONFIG)
output_producer = KafkaProducer(config=OUTPUT_PRODUCER_CONFIG)
scribe_producer = KafkaProducer(config=SCRIBE_PRODUCER_CONFIG)
strategy = AtlasStrategy()

step = AtlasStampClassifierStep(
    consumer,
    producer=output_producer,
    scribe_producer=scribe_producer,
    config=STEP_CONFIG,
    strategy=strategy,
    level=level,
)
step.start()
