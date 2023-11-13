from fastavro import schema, utils
from fastavro.repository.base import SchemaRepositoryError


def test_load_schema():
    try:
        loaded = schema.load_schema("schema.avsc")
    except SchemaRepositoryError:
        loaded = schema.load_schema("prv_candidates_step/schema.avsc")
    assert isinstance(loaded, dict)
    assert "aid" == loaded["fields"][0]["name"]
    assert "candid" == loaded["fields"][1]["name"]
    assert "detections" == loaded["fields"][2]["name"]
    assert "non_detections" == loaded["fields"][3]["name"]


def test_data():
    try:
        loaded = schema.load_schema("schema.avsc")
    except SchemaRepositoryError:
        loaded = schema.load_schema("prv_candidates_step/schema.avsc")
    data = utils.generate_one(loaded)
    assert "detections" in data
    assert "non_detections" in data
    assert "candid" in data
