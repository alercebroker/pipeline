from fastavro import schema, utils


def test_load_schema():
    loaded = schema.load_schema("schema.avsc")
    assert isinstance(loaded, dict)
    assert "aid" == loaded["fields"][0]["name"]
    assert "detections" == loaded["fields"][1]["name"]
    assert "non_detections" == loaded["fields"][2]["name"]


def test_data():
    loaded = schema.load_schema("schema.avsc")
    data = utils.generate_one(loaded)
    assert "detections" in data
    assert "non_detections" in data
