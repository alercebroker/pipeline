from pandas import DataFrame
from lc_classification.core.parsers.elasticc_parser import ElasticcParser
from alerce_classifiers.base.dto import OutputDTO

probabilities = DataFrame(
    {
        "Ia": ["0.5", "0.5", "0.5"],
        "AGN": ["0.4", "0.4", "0.4"],
        "Meta/Other": ["0.1", "0.1", "0.1"],
        "oid": ["oid1", "oid2", "oid3"],
    }
)
probabilities.set_index("oid", inplace=True)
hierarchical = {"top": DataFrame(), "children": {}}
output_dto = OutputDTO(probabilities, hierarchical)

messages = [
    {
        "detections": [
            {
                "oid": "oid1",
                "aid": "11",
                "candid": 1,
                "mjd": 11,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            },
            {
                "oid": "oid1",
                "aid": "1",
                "candid": 22,
                "mjd": 22,
                "new": False,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            },
        ],
        "non_detections": [],
        "oid": "oid1",
    },
    {
        "detections": [
            {
                "oid": "oid2",
                "aid": "2",
                "candid": 2,
                "mjd": 2,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "oid": "oid2",
    },
    {
        "detections": [
            {
                "oid": "oid3",
                "aid": "3",
                "candid": 3,
                "mjd": 3,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "oid": "oid3",
    },
    {
        "detections": [
            {
                "oid": "oid4",
                "aid": "4",
                "candid": 4,
                "mjd": 4,
                "new": True,
                "has_stamp": True,
                "extra_fields": {
                    "surveyPublishTimestamp": 1,
                    "brokerIngestTimestamp": 1,
                },
            }
        ],
        "non_detections": [],
        "oid": "oid4",
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
        filter(
            lambda x: x["classId"] == 300, result.value[-1]["classifications"]
        )
    )
    assert len(noclass) == 1
    assert noclass[0]["probability"] == 1


def test_parse_without_new_detections():
    parser = ElasticcParser()
    for msg in messages:
        msg["detections"][0]["new"] = False
    result = parser.parse(
        output_dto,
        messages=messages,
        classifier_version="test",
        classifier_name="test",
    )
    assert len(result.value) == len(messages)
    assert result.value[0]["alertId"] == 22
