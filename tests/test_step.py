import unittest
import logging
import sys
import os

from settings import *
PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)
from earlyclassifier import EarlyClassifier
from apf.consumers import AVROFileConsumer


class TestStep(unittest.TestCase):
    def test_is_responding_from_avros(self):
        level = logging.INFO
        logging.basicConfig(level=level,
                            format='%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', )

        consumer = AVROFileConsumer(config=CONSUMER_CONFIG)
        step = EarlyClassifier(consumer, config=STEP_CONFIG, level=level)
        step.start()


if __name__ == '__main__':
    unittest.main()
