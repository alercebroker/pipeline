from .surveyDataJoiner import SurveyDataJoiner
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ZTFDataJoiner(SurveyDataJoiner):
    """ZTF-specific data joining strategy."""
    #TODO FILL THIS OUT