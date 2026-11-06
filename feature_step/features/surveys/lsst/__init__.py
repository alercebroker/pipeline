"""lsst survey: constants, preprocessor and extractor."""
from features.surveys.base import Survey, SurveyBands  # contract and band table


class LSSTSurvey(Survey):
    """LSST survey definition."""

    name = "lsst"  # survey name
    sid = 1  # survey id
    tid = 1  # type id
    needs_references = False  # lsst does not read references
    bands = SurveyBands(  # band table of lsst
        {"u": 6, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5},  # single bands
        {"u,g": 61, "g,r": 12, "r,i": 23, "i,z": 34, "z,y": 45},  # band pairs
    )  # end of band table

    def make_preprocessor(self):
        """Return the lsst lightcurve preprocessor."""
        from lc_classifier.features.preprocess.lsst import LSSTLightcurvePreprocessor  # lazy heavy import

        return LSSTLightcurvePreprocessor()  # same arguments as step.py today

    def make_extractor(self):
        """Return the lsst feature extractor."""
        from lc_classifier.features.composites.lsst import LSSTFeatureExtractor  # lazy heavy import

        return LSSTFeatureExtractor()  # same arguments as step.py today
