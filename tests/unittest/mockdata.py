valid_data_dict = {
    "collection": "object",
    "type": "insert",
    "criteria": {"_id": "AID51423"},
    "data": {
        "field1": "some_field",
        "field2": "some_other_field",
    },
    "options": {}
}

valid_probabilities_dict = {
    "collection": "object",
    "type": "insert_probabilities",
    "criteria": {"_id": "AID51423"},
    "data": {
        "classifier_name": "classifier",
        "classifier_version": "1.0.0",
        "class1": 0.3,
        "class2": 0.7,
    },
    "options": {}
}

valid_data_json = """
{
    "collection": "object",
    "type": "insert",
    "criteria": {"_id": "AID51423"},
    "data": {"field1": "some_field", "field2": "some_other_field"}
}
"""
