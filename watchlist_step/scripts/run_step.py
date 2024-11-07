import logging
import os
import sys

import settings
from watchlist_step.step import WatchlistStep

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))

sys.path.append(PACKAGE_PATH)

level = logging.INFO
if "LOGGING_DEBUG" in locals():
    if settings.LOGGING_DEBUG:
        level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# if "CLASS" in settings.CONSUMER_CONFIG:
#     Consumer = get_class(settings.CONSUMER_CONFIG["CLASS"])
# else:
#     from apf.consumers import KafkaConsumer as Consumer
#
# consumer = Consumer(config=settings.CONSUMER_CONFIG)

step = WatchlistStep(
    config=settings.STEP_CONFIG,
    level=level,
    strategy_name=settings.UPDATE_STRATEGY,
)
step.start()
