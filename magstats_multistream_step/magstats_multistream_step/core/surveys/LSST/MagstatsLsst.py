from typing import List

import numpy as np
import pandas as pd

from ...BaseStatistics.BaseMagstats import BaseMagnitudeStatistics
from ...StatisticsSelector.statistics_selector import register_survey_class_magstat

class LSSTMagnitudeStatistics(BaseMagnitudeStatistics):
    """Generic magnitude statistics - only base stats"""
    pass  # do nothing

register_survey_class_magstat("LSST", LSSTMagnitudeStatistics)