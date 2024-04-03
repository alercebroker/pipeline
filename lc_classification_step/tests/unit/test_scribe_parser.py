from pandas import DataFrame
from alerce_classifiers.base.dto import OutputDTO
from lc_classification.core.parsers.scribe_parser import ScribeParser
from lc_classification.core.parsers.kafka_parser import KafkaOutput

probabilities = DataFrame(
    {
        "SN": ["0.5", "0.5", "0.5"],
        "AGN": ["0.4", "0.4", "0.4"],
        "Other": ["0.1", "0.1", "0.1"],
        "aid": ["aid1", "aid2", "aid3"],
    }
)
probabilities.set_index("aid", inplace=True)
hierarchical = {"top": DataFrame(), "children": {}}
output_dto = OutputDTO(probabilities, hierarchical)


def test_parse():
    parser = ScribeParser(classifier_name="test_classifier")
    result: KafkaOutput = parser.parse(
        output_dto, classifier_version="test_v1", oids={}
    )
    for res in result.value:
        assert res["data"]["classifier_version"] == "test_v1"
        assert res["collection"] == "object"
        assert res["type"] == "update_probabilities"
        assert "_id" in res["criteria"]
        assert "classifier_name" in res["data"]
        assert res["data"]["SN"] == "0.5"
        assert res["data"]["AGN"] == "0.4"
        assert res["data"]["Other"] == "0.1"
