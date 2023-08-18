from pandas import DataFrame
from lc_classification.core.parsers.alerce_parser import AlerceParser
from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.input_dto import create_features_dto

probabilities = DataFrame(
    {
        "Ia": [0.5, 0.5, 0.5],
        "AGN": [0.4, 0.4, 0.4],
        "Meta/Other": [0.1, 0.1, 0.1],
        "aid": ["aid1", "aid2", "aid3"],
        "classifier_name": ["test", "test", "test"],
    }
)
probabilities.set_index("aid", inplace=True)
hierarchical = {
    "top": DataFrame(
        {
            "aid": ["aid1", "aid2", "aid3"],
            "Periodic": [0.434, 0.434, 0.434],
            "Stochastic": [0.21, 0.21, 0.21],
            "Transient": [0.356, 0.356, 0.356],
        }
    ),
    "children": {
        "Transient": DataFrame(
            {
                "aid": ["aid1", "aid2", "aid3"],
                "Ia": [0.5, 0.5, 0.5],
            }
        ),
        "Stochastic": DataFrame(
            {
                "aid": ["aid1", "aid2", "aid3"],
                "AGN": [0.4, 0.4, 0.4],
            }
        ),
        "Periodic": DataFrame(
            {
                "aid": ["aid1", "aid2", "aid3"],
                "Meta/Other": [0.1, 0.1, 0.1],
            }
        ),
    },
}
output_dto = OutputDTO(probabilities, hierarchical)

messages = [
    {
        "detections": [
            {
                "aid": "aid1",
                "oid": 1,
                "candid": 1,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "aid": "aid1",
        "features": {"feature1": 1},
    },
    {
        "detections": [
            {
                "aid": "aid2",
                "oid": 2,
                "candid": 2,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "aid": "aid2",
        "features": {"feature1": 1},
    },
    {
        "detections": [
            {
                "aid": "aid3",
                "oid": 3,
                "candid": 3,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "aid": "aid3",
        "features": {"feature1": 1},
    },
]


def test_parse():
    parser = AlerceParser()
    features = create_features_dto(messages)
    result = parser.parse(
        output_dto,
        messages=messages,
        classifier_version="test",
        classifier_name="test",
        features=features,
    )
    for res in result.value:
        assert res["features"] == {"feature1": 1}
        assert "probabilities" in res["lc_classification"]
        assert "hierarchical" in res["lc_classification"]
        assert "top" in res["lc_classification"]["hierarchical"]
        assert "children" in res["lc_classification"]["hierarchical"]
        assert "Periodic" in res["lc_classification"]["hierarchical"]["top"]
        assert "Stochastic" in res["lc_classification"]["hierarchical"]["top"]
        assert "Transient" in res["lc_classification"]["hierarchical"]["top"]
        assert "Periodic" in res["lc_classification"]["hierarchical"]["children"]
        assert "Stochastic" in res["lc_classification"]["hierarchical"]["children"]
        assert "Transient" in res["lc_classification"]["hierarchical"]["children"]
