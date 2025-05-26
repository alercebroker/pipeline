from db_plugins.db.mongo.models import Object
from unittest import mock
from db_plugins.db.mongo.helpers.update_probs import (
    create_or_update_probabilities,
    create_or_update_probabilities_bulk,
)
from db_plugins.db.mongo._connection import MongoConnection
import unittest
import mongomock

"""
This test the object creation (not the main objetive)
and explain how to query and update the objects with
probabilities (main focus of the tests)
"""


class TestMongoProbabilities(unittest.TestCase):
    @mock.patch("db_plugins.db.mongo._connection.MongoClient")
    def setUp(self, mock_mongo):
        mock_mongo.return_value = mongomock.MongoClient()
        self.config = {
            "HOST": "host",
            "USERNAME": "username",
            "PASSWORD": "pwd",
            "PORT": 27017,
            "DATABASE": "database",
        }
        self.mongo_connection = MongoConnection(config=self.config)
        self.mongo_connection.create_db()
        self.database = self.mongo_connection.database
        self.obj_collection = self.database["object"]

    def create_2_objects(self):
        model_1 = Object(
            _id="oid1",
            aid="aid1",
            sid="sid",
            tid="tid1",
            corrected=False,
            stellar=False,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1.0,
            meanra=100.0,
            sigmara=0.1,
            meandec=50.0,
            sigmadec=0.2,
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
        model_2 = Object(
            _id="oid2",
            aid="aid2",
            sid="sid",
            tid="tid2",
            corrected=False,
            stellar=False,
            sigmara=0.2,
            sigmadec=0.1,
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1.0,
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

    def test_create_probabilities(self):
        self.create_2_objects()

        create_or_update_probabilities(
            self.mongo_connection.database,
            "stamp_classifier",
            "stamp_classifier_1.0.0",
            "oid2",
            {
                "CLASS1": 0.3,
                "CLASS2": 0.7,
            },
        )

        f1 = self.obj_collection.find_one({"_id": "oid2"})

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

        create_or_update_probabilities(
            self.mongo_connection.database,
            "stamp_classifier",
            "stamp_classifier_1.0.0",
            "oid1",
            {
                "CLASS1": 0.3,
                "CLASS2": 0.7,
            },
        )

        f1 = self.obj_collection.find_one({"_id": "oid1"})

        # Mind that the update don't change the order
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

    def test_bulk_create_or_update_probabilities(self):
        self.create_2_objects()

        create_or_update_probabilities_bulk(
            self.mongo_connection.database,
            "stamp_classifier",
            "stamp_classifier_1.0.0",
            ["oid1", "oid2"],
            [
                {
                    "CLASS1": 0.3,
                    "CLASS2": 0.7,
                },
                {
                    "CLASS1": 0.8,
                    "CLASS2": 0.2,
                },
            ],
        )

        f1 = self.obj_collection.find_one({"_id": "oid1"})
        f2 = self.obj_collection.find_one({"_id": "oid2"})

        expected_probabilities_1 = [
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
        expected_probabilities_2 = [
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
                "class_name": "CLASS1",
                "probability": 0.8,
                "ranking": 1,
            },
            {
                "classifier_name": "stamp_classifier",
                "classifier_version": "stamp_classifier_1.0.0",
                "class_name": "CLASS2",
                "probability": 0.2,
                "ranking": 2,
            },
        ]

        self.assertEqual(f1["probabilities"], expected_probabilities_1)
        self.assertEqual(f2["probabilities"], expected_probabilities_2)
