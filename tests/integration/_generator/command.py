import json
from random import seed, randint
from typing import Dict


class CommandGenerator:
    def __init__(self) -> None:
        seed(0)
        self._generated_inserts = 0
        self._command_hash: Dict[str, dict] = {
            "insert": {},
            "update": {},
            "update_probability": {},
            "update_feature": {},
        }
        self._generate_insert()

    def _generate_insert(self):
        command = {
            "collection": "object",
            "type": "insert",
            "data": {
                "_id": f"ID{self._generated_inserts}",
                "field1": "original",
                "field2": "original",
            },
            "options": {},
        }
        self._command_hash["insert"][self._generated_inserts] = command["data"]
        self._command_hash["update"][self._generated_inserts] = {}
        self._command_hash["update_probability"][self._generated_inserts] = {}
        self._command_hash["update_feature"][self._generated_inserts] = {}
        self._generated_inserts += 1
        return {"payload": json.dumps(command)}

    def _generate_update(self):
        aid = randint(0, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": f"ID{aid}"},
            "data": {"field1": "updated", "field3": "new_one"},
            "options": {},
        }
        self._command_hash["update"][aid].update(command["data"])
        return {"payload": json.dumps(command)}

    def _generate_update_probabilities(self):
        aid = randint(0, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update_probabilities",
            "criteria": {"_id": f"ID{aid}"},
            "data": {
                "classifier_name": "classifier",
                "classifier_version": "1.0.0",
                "class1": 0.3,
                "class2": 0.7,
            },
            "options": {},
        }
        self._command_hash["update_probability"][aid].update(command["data"])
        return {"payload": json.dumps(command)}

    def _generate_update_features(self):
        aid = randint(0, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": f"ID{aid}"},
            "data": {
                "features_version": "v1",
                "features": [
                    {"name": "feature1", "value": 12.34, "fid": 0},
                    {"name": "feature2", "value": None, "fid": 2},
                ],
            },
            "options": {},
        }
        self._command_hash["update_feature"][aid].update(command["data"])
        return {"payload": json.dumps(command)}

    def generate_random_command(self):
        dice_roll = randint(0, 99)
        if dice_roll in range(0, 30):
            return self._generate_insert()
        if dice_roll in range(30, 50):
            return self._generate_update()
        if dice_roll in range(50, 75):
            return self._generate_update_probabilities()
        return self._generate_update_features()
