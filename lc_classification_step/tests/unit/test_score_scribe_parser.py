from pandas import DataFrame
from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.scribe_parser import ScoreScribeParser
from lc_classification.core.parsers.kafka_parser import KafkaOutput

probabilities = DataFrame(
    {
        "Periodic": ["5", "5", "5"],
        "Stochastic": ["4", "4", "4"],
        "Transient": ["10", "10", "10"],
        "oid": ["oid1", "oid2", "oid3"],
    }
)
probabilities.set_index("oid", inplace=True)
hierarchical = {"top": DataFrame(), "children": {}}
output_dto = OutputDTO(probabilities, hierarchical)


def test_parse():
    parser = ScoreScribeParser(classifier_name="test_detector")
    result: KafkaOutput = parser.parse(
        output_dto, classifier_version="1.0.0", oids={}
    )
    for res in result.value:
        assert res["data"]["detector_version"] == "1.0.0"
        assert res["collection"] == "score"
        assert res["type"] == "insert"
        assert "_id" in res["criteria"]
        assert res["criteria"]["_id"] in ["oid1", "oid2", "oid3"]
        assert "test_detector" in res["data"]["detector_name"]
        assert {"name": "Periodic", "score": "5"} in res["data"]["categories"]
        assert {"name": "Stochastic", "score": "4"} in res["data"]["categories"]
        assert {"name": "Transient", "score": "10"} in res["data"]["categories"]
