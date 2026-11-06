"""ztf band table, in its own module so the ztf submodules can use it without importing ZTFSurvey."""
from features.surveys.base import SurveyBands  # band table type

ZTF_BANDS = SurveyBands({"g": 1, "r": 2}, {"g,r": 12}, letters={"g": 1, "r": 2, "i": 3})  # i only has a letter
