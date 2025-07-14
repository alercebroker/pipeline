from .core.StatisticsSelector.statistics_selector import get_object_statistics_class, get_magnitude_statistics_class, survey_registry
from .core.BaseStatistics.BaseMagstats import BaseMagnitudeStatistics
from .core.BaseStatistics.BaseObjectstats import BaseObjectStatistics
from .core.BaseStatistics._base import BaseStatistics
from .step import MagstatsStep_Multistream

#from . import surveys

__all__ = [
    "MagstatsStep",
    "BaseStatistics", 
    "BaseObjectStatistics", 
    "BaseMagnitudeStatistics",
    "get_object_statistics_class",
    "get_magnitude_statistics_class",
    "survey_registry"
]
