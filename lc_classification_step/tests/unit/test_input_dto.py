from lc_classification.core.parsers.input_dto import create_detections_dto
import pickle


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
