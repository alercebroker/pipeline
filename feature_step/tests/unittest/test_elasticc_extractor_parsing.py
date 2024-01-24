from features.core.elasticc import ELAsTiCCFeatureExtractor
from lc_classifier.features.preprocess.preprocess_elasticc import (
    ElasticcPreprocessor,
)
from lc_classifier.features.custom.elasticc_feature_extractor import (
    ElasticcFeatureExtractor,
)
from tests.data.elasticc_message_factory import generate_alert
from pandas import DataFrame


def test_create_lightcurve_dataframe():
    preprocessor = ElasticcPreprocessor(stream=True)
    lc_classifier_extractor = ElasticcFeatureExtractor(round=2)
    detections = generate_alert(
        oid="oid1", band="u", num_messages=1, identifier="1"
    )
    step_extractor = ELAsTiCCFeatureExtractor(
        preprocessor, lc_classifier_extractor, detections
    )
    lightcurve_dataframe = step_extractor._create_lightcurve_dataframe(
        detections
    )
    assert len(lightcurve_dataframe.loc["oid1"]) == 25
    assert lightcurve_dataframe.iloc[0]["aid"] == "aid1"
    assert lightcurve_dataframe.iloc[0]["BAND"] == "u"
    assert lightcurve_dataframe.iloc[0]["FLUXCAL"] >= 15
    assert lightcurve_dataframe.iloc[0]["FLUXCAL"] <= 20
    assert lightcurve_dataframe.iloc[0]["FLUXCALERR"] >= 0
    assert lightcurve_dataframe.iloc[0]["FLUXCALERR"] <= 1
    assert lightcurve_dataframe.iloc[0]["MJD"] >= 59000
    assert lightcurve_dataframe.iloc[0]["MJD"] <= 60000


def test_create_metadata_dataframe():
    preprocessor = ElasticcPreprocessor(stream=True)
    lc_classifier_extractor = ElasticcFeatureExtractor(round=2)
    detections = generate_alert(
        oid="oid1", band="u", num_messages=1, identifier="1"
    )
    step_extractor = ELAsTiCCFeatureExtractor(
        preprocessor, lc_classifier_extractor, detections
    )
    lightcurve_dataframe = step_extractor._create_lightcurve_dataframe(
        detections
    )
    result = preprocessor.preprocess(lightcurve_dataframe)
    metadata_dataframe = step_extractor._create_metadata_dataframe(result)
    # there are 64 metadata columns
    assert metadata_dataframe.shape == (1, 64)


def test_preprocessor_can_run_with_parsed_data():
    preprocessor = ElasticcPreprocessor(stream=True)
    lc_classifier_extractor = ElasticcFeatureExtractor(round=2)
    detections = generate_alert(
        oid="oid1", band="u", num_messages=1, identifier="1"
    )
    step_extractor = ELAsTiCCFeatureExtractor(
        preprocessor, lc_classifier_extractor, detections
    )
    lightcurve_dataframe = step_extractor._create_lightcurve_dataframe(
        detections
    )
    result = preprocessor.preprocess(lightcurve_dataframe)
    print(result)
    assert isinstance(result, DataFrame)
