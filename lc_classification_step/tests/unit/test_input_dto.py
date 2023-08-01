from lc_classification.core.parsers.input_dto import create_detections_dto, create_features_dto
import pickle
import numpy as np


def test_create_detections_dto():
    def extra_fields():
        return {"ef1": pickle.dumps([{"data1": "data1"}]), "ef2": "val2"}

    messages = [
        {
            "detections": [
                {"aid": "aid1", "candid": "cand1", "extra_fields": extra_fields()},
                {"aid": "aid1", "candid": "cand2", "extra_fields": extra_fields()},
            ]
        },
        {
            "detections": [
                {"aid": "aid2", "candid": "cand3", "extra_fields": {}},
            ]
        },
    ]
    detections = create_detections_dto(messages)
    assert detections.index.tolist() == ["aid1", "aid2"]
    assert list(detections["extra_fields"].values) == [
        {"ef1": {"data1": "data1"}, "ef2": "val2"},
        {},
    ]

def test_create_features_dto():
    messages = [
            {
                'aid': 'aid1',
                'features': {'feat1': 1, 'feat2': 2}
            },
            {
                'aid': 'aid1',
                'features': {'feat1': 2, 'feat2': 3}
            },
            {
                'aid': 'aid2',
                'features': {'feat1': 4, 'feat2': 5}
            },
            {
                'aid': 'aid3',
            },
            {
                'aid': 'aid4',
                'features': {'feat1': 4}
            },
        ]
    features = create_features_dto(messages)
    assert features.loc['aid1', 'feat1'] == 2
    assert features.loc['aid1', 'feat2'] == 3
    assert np.isnan(features.loc['aid4', 'feat2'])
    assert features.index.tolist() == ["aid1", "aid2", "aid4"]
