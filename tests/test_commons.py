def assert_object_is_correct(obj):
    assert "aid" in obj
    assert "features" in obj
    assert "lc_classification" in obj


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
        "brokerPublishTimestamp",
        "brokerName",
    ]
    for field in valid_fields:
        assert field in obj
        
    assert isinstance(obj["classifications"], list)
    assert len(obj["classifications"]) > 0


def assert_command_is_correct(command):
    assert command["collection"] == "object"
    assert command["type"] == "update_probabilities"
    assert command["criteria"]["_id"] is not None
    assert "aid" not in command["data"]
    assert not command["options"]["set_on_insert"]