from typing import Union, Literal

import pandas as pd
from ...BaseStatistics.BaseObjectstats import BaseObjectStatistics
from ...StatisticsSelector.statistics_selector import register_survey_class_objstat

class LSSTObjectStatistics(BaseObjectStatistics):
    """LSST object statistics - only base stats"""
    pass  # do nothing

register_survey_class_objstat("LSST", LSSTObjectStatistics)