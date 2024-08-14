from pytest import approx


def assert_ztf_object_is_correct(obj):
    assert "oid" in obj
    assert "features" in obj
    assert "lc_classification" in obj
    assert "class" in obj["lc_classification"]


def assert_elasticc_object_is_correct(obj):
    valid_fields = [
        "alertId",
        "diaSourceId",
        "elasticcPublishTimestamp",
        "brokerIngestTimestamp",
        "classifications",
        "brokerVersion",
        "classifierName",
        "classifierParams",
        "brokerName",
    ]
    for field in valid_fields:
        assert field in obj

    assert isinstance(obj["classifications"], list)
    assert len(obj["classifications"]) > 0
    suma = 0
    for classification in obj["classifications"]:
        suma += classification["probability"]
    assert suma == approx(1)


def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert not command["options"]["set_on_insert"]


def assert_score_command_is_correct(command):
    assert command["collection"] == "score"
    assert command["type"] == "insert"
    assert command["criteria"]["_id"] is not None
    assert not command["options"]["set_on_insert"]
