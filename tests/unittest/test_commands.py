import unittest

from pymongo.operations import InsertOne, UpdateOne

from mongo_scribe.command.exceptions import NoFearuteVersionProvidedException

from mongo_scribe.command.commands import (
    InsertCommand,
    UpdateCommand,
    UpdateProbabilitiesCommand,
    UpdateFeaturesCommand,
)
from mongo_scribe.command.exceptions import (
    NoDataProvidedException,
    UpdateWithNoCriteriaException,
    NoCollectionProvidedException,
)

from mockdata import (
    valid_data_dict,
    valid_probabilities_dict,
    valid_features_dict,
)


class CommandTests(unittest.TestCase):
    def test_create_dbcommand(self):
        command = InsertCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        self.assertEqual(command.data, valid_data_dict["data"])

    def test_create_dbcommand_no_collection(self):
        self.assertRaises(
            NoCollectionProvidedException,
            lambda: InsertCommand(None, valid_data_dict["data"], None),
        )

    def test_create_dbcommand_no_data(self):
        self.assertRaises(
            NoDataProvidedException,
            lambda: InsertCommand("object", None, None),
        )

    def test_create_dbcommand_update_without_criteria(self):
        self.assertRaises(
            UpdateWithNoCriteriaException,
            lambda: UpdateCommand(
                "object", "update", None, valid_data_dict["data"]
            ),
        )

    def test_insert_dbcommand_get_operation(self):
        insert_command = InsertCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        operations = insert_command.get_operations()
        self.assertEqual(len(operations), 1)
        self.assertIsInstance(operations[0], InsertOne)

    def test_insert_dbcommand_get_operation_has_data_as_document(self):
        insert_command = InsertCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        operations = insert_command.get_operations()
        self.assertEqual(operations[0]._doc, valid_data_dict["data"])

    def test_update_dbcommand_get_operation(self):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        operations = update_command.get_operations()
        self.assertEqual(len(operations), 1)
        self.assertIsInstance(operations[0], UpdateOne)

    def test_update_get_operation_without_set_on_insert_has_data_with_set_as_document(
        self,
    ):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        operations = update_command.get_operations()
        self.assertEqual(operations[0]._filter, valid_data_dict["criteria"])
        self.assertEqual(operations[0]._doc, {"$set": valid_data_dict["data"]})

    def test_update_get_operation_with_set_on_insert_has_data_with_set_on_insert_as_document(
        self,
    ):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
            {"set_on_insert": True},
        )
        operations = update_command.get_operations()
        self.assertEqual(operations[0]._filter, valid_data_dict["criteria"])
        self.assertEqual(
            operations[0]._doc, {"$setOnInsert": valid_data_dict["data"]}
        )

    def test_update_db_command_options(self):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
            {"upsert": True},
        )
        operations = update_command.get_operations()
        self.assertTrue(operations[0]._upsert)

    def test_update_probabilities_with_set_on_insert_get_operation(self):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
            {"set_on_insert": True},
        )
        operations = update_command.get_operations()
        self.assertEqual(len(operations), 2)
        self.assertTrue(all(isinstance(op, UpdateOne) for op in operations))
        self.assertFalse(any(op._upsert for op in operations))

    def test_update_probabilities_with_set_on_insert_get_operation_has_data_with_push_as_document(
        self,
    ):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
            {"set_on_insert": True},
        )
        operations = update_command.get_operations()
        for op in operations:  # All include AID as criteria
            self.assertTrue(
                all(
                    pair in op._filter.items()
                    for pair in valid_probabilities_dict["criteria"].items()
                )
            )

        self.assertEqual(
            operations[0]._doc, {"$setOnInsert": {"probabilities": []}}
        )
        insert = (
            operations[1]._doc.pop("$push").pop("probabilities").pop("$each")
        )
        for element in insert:
            self.assertEqual(
                {
                    "probability",
                    "ranking",
                    "classifier_name",
                    "classifier_version",
                    "class_name",
                },
                set(element),
            )
        self.assertEqual(operations[1]._doc, {})

    def test_insert_probabilities_with_set_on_insert_options(self):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
            {"upsert": True, "set_on_insert": True},
        )
        operations = update_command.get_operations()
        self.assertTrue(operations[0]._upsert)
        self.assertFalse(operations[1]._upsert)

    def test_update_probabilities_get_operation(self):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
        )
        operations = update_command.get_operations()
        self.assertEqual(len(operations), 4)
        self.assertTrue(all(isinstance(op, UpdateOne) for op in operations))
        self.assertFalse(any(op._upsert for op in operations))

    def test_update_probabilities_get_operation_has_data_with_set_as_document(
        self,
    ):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
        )
        operations = update_command.get_operations()
        for op in operations:  # All include AID as criteria
            self.assertTrue(
                all(
                    pair in op._filter.items()
                    for pair in valid_probabilities_dict["criteria"].items()
                )
            )

        self.assertEqual(
            operations[0]._doc, {"$setOnInsert": {"probabilities": []}}
        )
        insert = (
            operations[1]._doc.pop("$push").pop("probabilities").pop("$each")
        )
        for element in insert:
            self.assertEqual(
                {
                    "probability",
                    "ranking",
                    "classifier_name",
                    "classifier_version",
                    "class_name",
                },
                set(element),
            )
        self.assertEqual(operations[1]._doc, {})
        for op in operations[2:]:
            self.assertTrue(
                all(
                    pair in op._filter.items()
                    for pair in valid_probabilities_dict["criteria"].items()
                )
            )
            update = op._doc.pop("$set")
            self.assertEqual(
                {
                    "probabilities.$[el].probability",
                    "probabilities.$[el].ranking",
                },
                set(update),
            )
            self.assertEqual(op._doc, {})

    def test_update_probabilities_options(self):
        update_command = UpdateProbabilitiesCommand(
            valid_probabilities_dict["collection"],
            valid_probabilities_dict["data"].copy(),
            valid_probabilities_dict["criteria"],
            {"upsert": True},
        )
        operations = update_command.get_operations()
        self.assertTrue(operations[0]._upsert)
        self.assertFalse(any(op._upsert for op in operations[1:]))

    def test_update_db_command_default_options(self):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
        )
        operations = update_command.get_operations()
        self.assertFalse(operations[0]._upsert)

    def test_update_db_command_unsupported_options(self):
        update_command = UpdateCommand(
            valid_data_dict["collection"],
            valid_data_dict["data"],
            valid_data_dict["criteria"],
            {"hehe": "hehe"},
        )
        operations = update_command.get_operations()
        self.assertFalse(operations[0]._upsert)

# test for update features

    def test_update_features_check_input_wrong_input(self):
        invalid_features_data = {
            "feat1": 123,
            "feat2": 321,
        }
        with self.assertRaises(NoFearuteVersionProvidedException):
            update_features_command = UpdateFeaturesCommand(
                collection=valid_features_dict["collection"],
                data=invalid_features_data,
                criteria=valid_features_dict["criteria"],
                options=valid_features_dict["options"],
            )

    def test_update_features_with_set_on_insert(self):
        update_features_options = {"set_on_insert": True}
        update_features_command = UpdateFeaturesCommand(
            collection=valid_features_dict["collection"],
            data=valid_features_dict["data"].copy(),
            criteria=valid_features_dict["criteria"],
            options=update_features_options,
        )

        operations = update_features_command.get_operations()

        self.assertEqual(len(operations), 2)
        self.assertEqual(
            operations[0]._doc, {"$setOnInsert": {"features": []}}
        )
        self.assertEqual(operations[0]._filter, {"_id": "AID51423"})
        self.assertFalse(operations[0]._upsert)
        self.assertEqual(
            operations[1]._doc,
            {
                "$push": {
                    "features": {
                        "$each": [
                            {
                                "features_version": "v1",
                                "feature_name": "feature1",
                                "feature_value": 12.34,
                            },
                            {
                                "features_version": "v1",
                                "feature_name": "feature2",
                                "feature_value": None,
                            },
                        ]
                    }
                }
            },
        )
        self.assertEqual(
            operations[1]._filter,
            {"features.features_version": {"$ne": "v1"}, "_id": "AID51423"},
        )

    def test_update_features_with_set_on_insert_with_upsert(self):
        update_features_options = {
            "set_on_insert": True,
            "upsert": True,
        }
        update_features_command = UpdateFeaturesCommand(
            collection=valid_features_dict["collection"],
            data=valid_features_dict["data"].copy(),
            criteria=valid_features_dict["criteria"],
            options=update_features_options,
        )

        operations = update_features_command.get_operations()

        self.assertEqual(len(operations), 2)
        self.assertEqual(
            operations[0]._doc, {"$setOnInsert": {"features": []}}
        )
        self.assertEqual(operations[0]._filter, {"_id": "AID51423"})
        self.assertTrue(operations[0]._upsert)

    def test_update_features_data_operations(self):
        update_features_command = UpdateFeaturesCommand(
            collection=valid_features_dict["collection"],
            data=valid_features_dict["data"].copy(),
            criteria=valid_features_dict["criteria"],
            options=valid_features_dict["options"],
        )

        operations = update_features_command.get_operations()

        self.assertEqual(len(operations), 4)
        self.assertEqual(operations[2]._filter, {"_id": "AID51423"})
        self.assertEqual(
            operations[2]._doc,
            {
                "$set": {
                    "features.$[el].feature_name": "feature1",
                    "features.$[el].feature_value": 12.34,
                }
            },
        )
        self.assertEqual(
            operations[2]._array_filters,
            [
                {
                    "el.features_version": "v1",
                }
            ],
        )
        self.assertEqual(operations[3]._filter, {"_id": "AID51423"})
        self.assertEqual(
            operations[3]._doc,
            {
                "$set": {
                    "features.$[el].feature_name": "feature2",
                    "features.$[el].feature_value": None,
                }
            },
        )
        self.assertEqual(
            operations[3]._array_filters,
            [
                {
                    "el.features_version": "v1",
                }
            ],
        )
