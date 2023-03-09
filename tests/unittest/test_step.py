import unittest
import pytest
import pandas as pd
import numpy as np
from unittest import mock

from apf.producers import KafkaProducer
from magstats_step.utils.multi_driver.connection import MultiDriverConnection
from magstats_step.step import MagstatsStep

from data.messages import *

class StepTestCase(unittest.TestCase):
    def setUp(self) -> None:
        step_config = {
        }
        mock_producer = mock.create_autospec(KafkaProducer)
        self.step = MagstatsStep(
            config=step_config,
        )

    def tearDown(self) -> None:
        del self.step

    def test_step(self):
        self.step.execute(LC_MESSAGE)
        # Verify magstats insert call
        dict_to_insert = pd.DataFrame.from_dict(MAGSTATS_RESULT).replace({np.nan: None}).to_dict("records")
        return

    def test_lightcurve_parse(self):
        pass

