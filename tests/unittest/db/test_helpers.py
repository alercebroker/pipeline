from db_plugins.db.mongo.models import Object
from db_plugins.db.mongo.helpers.update_probs import create_of_update_probabilities
from db_plugins.db.mongo.connection import MongoConnection
import unittest
import mongomock

"""
This test the object creation (not the main objetive)
and explain how to query and update the objects with 
probabilities (main focus of the tests)
"""


class MongoProbabilitiesTest(unittest.TestCase):
    def setUp(self):
        client = mongomock.MongoClient()
        self.database = client["database"]
        self.obj_collection = self.database["object"]
        self.config = {
            "HOST": "host",
            "USERNAME": "username",
            "PASSWORD": "pwd",
            "PORT": 27017,
            "DATABASE": "database",
        }
        self.mongo_connection = MongoConnection(client=client)
        self.mongo_connection.connect(config=self.config)
        self.mongo_connection.create_db()

    def create_2_objects(self):
        model_1 = Object(
            aid="aid1",
            oid="oid1",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet=2,
            probabilities=[
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.6,
                    "ranking": 1,
                },
                {
                    "classifier_name": "stamp_classifier",
                    "classifier_version": "stamp_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.4,
                    "ranking": 2,
                },
            ],
        )
        model_2 = Object(
            aid="aid2",
            oid="oid2",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra=100.0,
            meandec=50.0,
            ndet=5,
            probabilities=[
                {
                    "classifier_name": "lc_classifier",
                    "classifier_version": "lc_classifier_1.0.0",
                    "class_name": "CLASS1",
                    "probability": 0.4,
                    "ranking": 2,
                },
                {
                    "classifier_name": "lc_classifier",
                    "classifier_version": "lc_classifier_1.0.0",
                    "class_name": "CLASS2",
                    "probability": 0.6,
                    "ranking": 1,
                },
            ],
        )
        self.obj_collection.insert_many([model_1, model_2])


    def test_error_input_length(self):
        with self.assertRaises(Exception):
            create_of_update_probabilities(self.mongo_connection, "aid1", "classifier1", "v1", ["class1", "class2"], [0.1, 0.2, 0.3])


    def test_create_probabilities(self):
        self.create_2_objects()

        create_of_update_probabilities(
            self.mongo_connection,
            "aid2",
            "stamp_classifier",
            "stamp_classifier_1.0.0",
            ["CLASS1", "CLASS2"],
            [0.3, 0.7]
        )

        f1 = self.obj_collection.find_one({"aid": "aid2"})

        expected_probabilities = [
            {
                "classifier_name": "lc_classifier",
                "classifier_version": "lc_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 0.4,
                "ranking": 2,
            },
            {
                "classifier_name": "lc_classifier",
                "classifier_version": "lc_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.6,
                "ranking": 1,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.7,
                "ranking": 1,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 0.3,
                "ranking": 2,
            },
        ]

        self.assertEqual(f1["probabilities"], expected_probabilities)

    def test_update_probabilities(self):
        self.create_2_objects()
        
        create_of_update_probabilities(
            self.mongo_connection,
            "aid1",
            "stamp_classifier",
            "stamp_classifier_1.0.0",
            ["CLASS1", "CLASS2"],
            [0.3, 0.7]
        )

        f1 = self.obj_collection.find_one({"aid": "aid1"})
        print(f"result \n {f1['probabilities']}")

        # Mind that the update dont change the order
        expected_probabilities = [
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS1",
                "probability": 0.3,
                "ranking": 2,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.7,
                "ranking": 1,
            },
        ]

        self.assertEqual(f1["probabilities"], expected_probabilities)
