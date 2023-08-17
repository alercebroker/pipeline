from pandas import DataFrame
from lc_classification.core.parsers.elasticc_parser import ElasticcParser
from alerce_classifiers.base.dto import OutputDTO

probabilities = DataFrame(
    {
        "Ia": ["0.5", "0.5", "0.5"],
        "AGN": ["0.4", "0.4", "0.4"],
        "Meta/Other": ["0.1", "0.1", "0.1"],
        "aid": ["aid1", "aid2", "aid3"],
    }
)
probabilities.set_index("aid", inplace=True)
hierarchical = {"top": DataFrame(), "children": {}}
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
    },
    {
        "detections": [
            {
                "aid": "aid4",
                "oid": 4,
                "candid": 4,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "aid": "aid4",
    },
]


def test_parse():
    parser = ElasticcParser()
    result = parser.parse(
        output_dto,
        messages=messages,
        classifier_version="test",
        classifier_name="test",
    )
    assert len(result.value) == 4
    for res in result.value:
        assert res["elasticcPublishTimestamp"] == 1
        assert res["brokerIngestTimestamp"] == 1
        assert len(res["classifications"]) > 0


def test_parse_no_class_probability_is_1():
    parser = ElasticcParser()
    result = parser.parse(
        output_dto,
        messages=messages,
        classifier_version="test",
        classifier_name="test",
    )
    noclass = list(
        filter(lambda x: x["classId"] == 300, result.value[-1]["classifications"])
    )
    assert len(noclass) == 1
    assert noclass[0]["probability"] == 1
