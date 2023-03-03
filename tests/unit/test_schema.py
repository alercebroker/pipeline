from fastavro import schema, utils


def test_load_schema():
    loaded = schema.load_schema("schema.avsc")
    assert isinstance(loaded, dict)
    assert "new_alert" == loaded["fields"][0]["name"]
    assert "prv_detections" == loaded["fields"][1]["name"]
    assert "non_detections" == loaded["fields"][2]["name"]


def test_data():
    loaded = schema.load_schema("schema.avsc")
    data = utils.generate_one(loaded)
    assert "new_alert" in data
    assert "prv_detections" in data
    assert "non_detections" in data
