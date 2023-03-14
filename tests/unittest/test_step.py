import unittest
import pytest
import pandas as pd
import numpy as np
from unittest import mock

from magstats_step.step import MagstatsStep

from data.messages import data

class StepTestCase(unittest.TestCase):
    def setUp(self) -> None:
        step_config = {
        }
        self.step = MagstatsStep(
            config=step_config,
        )

    def tearDown(self) -> None:
        del self.step

    def test_step(self):
        self.step.execute(data)
        pass

    def test_compute_magstats(self):
        stats = self.step.compute_magstats(data)
        assert 'aid' in stats[0]
        assert 'magstats' in stats[0]
        assert len(stats) == len(data)

