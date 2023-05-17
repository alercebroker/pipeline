import json
from random import seed, randint, choice
from typing import Dict

feature_versions = ["v1", "v2", "v3"]
classifiers = ["lc_classifier", "dummy_classifier", "random_classifier"]


# TODO:
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

    def clear(self):
        self._command_hash: Dict[str, dict] = {
            "insert": {},
            "update": {},
            "update_probability": {},
            "update_feature": {},
        }

    def set_offset(self, new_offset: int):
        self._generated_inserts = new_offset

    def get_updated_probabilities(self):
        return self._command_hash["update_probability"]

    def get_updated_features(self):
        return self._command_hash["update_feature"]

    def generate_insert(self, options={}):
        command = {
            "collection": "object",
            "type": "insert",
            "data": {
                "_id": f"ID{self._generated_inserts}",
                "field1": "original",
                "field2": "original",
            },
            "options": options,
        }
        self._command_hash["insert"][self._generated_inserts] = command["data"]
        self._command_hash["update"][self._generated_inserts] = {}
        self._command_hash["update_probability"][self._generated_inserts] = []
        self._command_hash["update_feature"][self._generated_inserts] = []
        self._generated_inserts += 1
        return {"payload": json.dumps(command)}

    def _add_feature(self, aid: int, feature: dict):
        def remove_value_from_dict(d: dict):
            return {k: v for k, v in d.items() if k != "value"}

        parsed_features = []
        for feats in feature["features"]:
            parsed_features.append(
                {**feats, "version": feature["features_version"]}
            )
        parsed_no_values = list(map(remove_value_from_dict, parsed_features))
        buffer: list = self._command_hash["update_feature"][aid]
        buffer = [
            feat
            for feat in buffer
            if remove_value_from_dict(feat) not in parsed_no_values
        ]
        buffer.extend(parsed_features)
        self._command_hash["update_feature"][aid] = buffer

    def _add_probability(self, aid: int, prob: dict):
        local_prob = prob.copy()
        parsed_probabilities = []
        classifier_name = local_prob.pop("classifier_name")
        classifier_version = local_prob.pop("classifier_version")
        for class_name, probability in local_prob.items():
            parsed_probabilities.append(
                {
                    "classifier_name": classifier_name,
                    "classifier_version": classifier_version,
                    "probability": probability,
                    "class_name": class_name,
                    "ranking": 1
                    if probability > 0.5
                    else 2,  # because probs are fixed
                }
            )
        buffer: list = self._command_hash["update_probability"][aid]
        buffer = [
            prob
            for prob in buffer
            if prob["classifier_name"] != classifier_name
            and prob["classifier_version"] != classifier_version
        ]
        buffer.extend(parsed_probabilities)
        self._command_hash["update_probability"][aid] = buffer

    def _generate_update(self, options={}, offset=0):
        aid = randint(offset, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update",
            "criteria": {"_id": f"ID{aid}"},
            "data": {"field1": "updated", "field3": "new_one"},
            "options": options,
        }
        self._command_hash["update"][aid].update(command["data"])
        return {"payload": json.dumps(command)}

    def _generate_update_probabilities(self, options={}, offset=0):
        aid = randint(offset, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update_probabilities",
            "criteria": {"_id": f"ID{aid}"},
            "data": {
                "classifier_name": choice(classifiers),
                "classifier_version": "1.0.0",
                "class1": 0.3,
                "class2": 0.7,
            },
            "options": options,
        }
        self._add_probability(aid, command["data"])
        return {"payload": json.dumps(command)}

    def _generate_update_features(self, options={}, offset=0):
        aid = randint(offset, self._generated_inserts - 1)
        command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": f"ID{aid}"},
            "data": {
                "features_version": choice(feature_versions),
                "features": [
                    {"name": "feature1", "value": 12.34, "fid": 0},
                    {"name": "feature2", "value": None, "fid": 2},
                ],
            },
            "options": options,
        }
        self._add_feature(aid, command["data"])
        return {"payload": json.dumps(command)}

    def generate_random_command(self, options={}, offset=0):
        dice_roll = randint(0, 99)
        if dice_roll in range(0, 30):
            return self.generate_insert(options)
        if dice_roll in range(30, 50):
            return self._generate_update(options, offset)
        if dice_roll in range(50, 75):
            return self._generate_update_probabilities(options, offset)
        return self._generate_update_features(options, offset)
