"""ztf survey: constants, preprocessor and extractor."""
from features.surveys.base import Survey, SurveyBands  # contract and band table


class ZTFSurvey(Survey):
    """ZTF survey definition."""

    name = "ztf"  # survey name
    sid = 0  # survey id
    tid = 0  # type id
    needs_references = True  # ztf reads references from the db
    bands = SurveyBands({"g": 1, "r": 2}, {"g,r": 12}, letters={"g": 1, "r": 2, "i": 3})  # i only has a letter

    def make_preprocessor(self):
        """Return the ztf lightcurve preprocessor."""
        from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor  # lazy heavy import

        return ZTFLightcurvePreprocessor(drop_bogus=True)  # same arguments as step.py today

    def make_extractor(self):
        """Return the ztf feature extractor."""
        from lc_classifier.features.composites.ztf import ZTFFeatureExtractor  # lazy heavy import

        return ZTFFeatureExtractor()  # same arguments as step.py today
