from pandas import DataFrame

from alerce_classifiers.base.factories import input_dto_factory

feat_dict = {}


def check_header_correct(preprocessed_input):
    assert feat_dict.values() in preprocessed_input.columns


def check_features_correct(preprocessed_input):
    # TODO: obtain list of features from the model class
    assert False


def test_preprocess():
    detections = DataFrame()
    features = DataFrame()
    dto = input_dto_factory(detections, None, features, None, None)
    mapper = BarneyMapper()
    check_header_correct(mapper.preprocess())
    check_features_correct(mapper.preprocess())


def test_postprocess():
    mapper = BarneyMapper()
    probabilities = DataFrame({"aid": ["aid1"], "SN": [1]})
    dto = mapper.postprocess(probabilities)
    assert dto.probabilities.aid.iloc[0] == probabilities.aid.iloc[0]
