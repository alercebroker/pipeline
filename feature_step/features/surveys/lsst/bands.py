"""lsst band table, in its own module so the lsst submodules can use it without importing LSSTSurvey."""
from features.surveys.base import SurveyBands  # band table type

LSST_BANDS = SurveyBands(  # band table of lsst
    {"u": 6, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},  # single bands
    {"u,g": 61, "g,r": 12, "r,i": 23, "i,z": 34, "z,y": 45},  # band pairs
)  # end of band table
